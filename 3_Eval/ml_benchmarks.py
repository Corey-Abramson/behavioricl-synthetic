"""
ML Benchmarks — BehaviorICL

OLS, Lasso, and Random Forest baselines against LLM ICL.

For each held-out trait:
  1. Load the same participants evaluated by ICL for that trait.
  2. Build an 8-feature matrix from the other traits (raw values, NaN for missing).
  3. Run 5-fold CV: OLS, Lasso (LassoCV inner alpha selection), Random Forest.
  4. Compare Pearson r and MAE against ICL, zero-shot, and PC1 baselines.

Notes on comparability:
  - ICL/zero-shot/PC1 are evaluated on the full participant set with no train/test
    split. ML models use 5-fold CV, so their Pearson r estimates out-of-sample
    performance and are not inflated by in-sample fitting.
  - PC1 is unsupervised (no target label used), so it is reported without CV.

Output:
    3_Eval/cross_trait/ml_benchmarks.txt

Usage:
    python 3_Eval/ml_benchmarks.py
"""

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "1_Prompts"))
from trait_configs import TRAIT_CONFIGS

TRAITS = [
    "trust", "cooperation", "coordination", "depth_of_reasoning",
    "tolerance_for_inequality", "risk_tolerance", "altruism", "morality",
    "competitiveness",
]

DATA_PATH = ROOT / "0_Preprocess" / "synthetic_data.csv"
SEED = 42
N_FOLDS = 5


# ── Feature extraction ────────────────────────────────────────────────────────

def _val(v):
    try:
        x = float(v)
        return None if math.isnan(x) else x
    except (TypeError, ValueError):
        return None


def get_all_features(row):
    """Return {trait: float | None} for all 9 traits from one CSV row."""
    f = {}

    f["trust"] = _val(row.get("trust.1.player.trust_send"))

    coop = [_val(row.get(f"cooperation.{r}.player.cooperate_decision")) for r in range(1, 5)]
    coop = [v for v in coop if v is not None]
    f["cooperation"] = sum(coop) / len(coop) if coop else None

    coord = [row.get(f"coordination.{r}.player.coordinate_decision") for r in range(1, 5)]
    coord = [1.0 if str(v) == "B" else 0.0 for v in coord if v is not None and str(v) in ("A", "B")]
    f["coordination"] = sum(coord) / len(coord) if coord else None

    f["depth_of_reasoning"] = _val(row.get("guessing_game.1.player.guess"))

    raw = _val(row.get("redistribute.1.player.redistribute_decision"))
    # slider stores B's credits; corrected A's credits = 10 - stored
    f["tolerance_for_inequality"] = (10 - raw) if raw is not None else None

    f["risk_tolerance"] = _val(row.get("games_solo.1.player.boxes_collected"))
    f["altruism"]       = _val(row.get("games_solo.1.player.dictator_decision"))

    moral = [_val(row.get(f"games_solo.1.player.moral_decision_{n}")) for n in range(1, 6)]
    moral = [v for v in moral if v is not None]
    f["morality"] = sum(moral) if moral else None

    f["competitiveness"] = _val(row.get("compete.1.player.compete_decision"))

    return f


# ── Data loaders ──────────────────────────────────────────────────────────────

def load_true_vals(trait_name):
    """Return {participant_code: true_value} from icl_results.jsonl."""
    config = TRAIT_CONFIGS[trait_name]
    path = ROOT / "2_Inference" / trait_name / "icl_results.jsonl"

    if trait_name == "trust":
        true_keys = ["true_trust_send"]
        direct = True
    else:
        true_keys = [f"true_{c.replace('.', '_')}" for c in config.target_cols]
        direct = False

    pairs = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        code = r["participant_code"]
        if direct:
            v = r.get(true_keys[0])
            if v is not None:
                pairs[code] = float(v)
        else:
            raw = [r.get(k) for k in true_keys]
            valid = []
            for v in raw:
                if v is None:
                    continue
                try:
                    valid.append(float(v))
                except (TypeError, ValueError):
                    pass
            if not valid:
                continue
            if trait_name == "morality":
                # count of donations, 0 to 5, the scale eval_trait.py scores on
                if len(valid) < len(true_keys):
                    continue
                t = sum(valid)
            else:
                t = sum(valid) / len(valid)
            if trait_name == "tolerance_for_inequality":
                t = 10 - t
            pairs[code] = t
    return pairs


def load_preds(trait_name, method):
    """Return {participant_code: predicted_value} for icl or zero_shot."""
    filenames = {
        "icl":       "icl_results.jsonl",
        "zero_shot": "zero_shot_results.jsonl",
    }
    path = ROOT / "2_Inference" / trait_name / filenames[method]
    config = TRAIT_CONFIGS[trait_name]
    pred_key = "pred_trust_send" if trait_name == "trust" else f"pred_{config.prediction_key}"
    result = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        p = r.get(pred_key)
        if p is not None:
            result[r["participant_code"]] = float(p)
    return result


# ── Statistics ────────────────────────────────────────────────────────────────

def pearson_r(xs, ys):
    a, b = np.asarray(xs, float), np.asarray(ys, float)
    if len(a) < 2:
        return float("nan")
    da, db = a - a.mean(), b - b.mean()
    num = (da * db).mean()
    denom = a.std() * b.std()
    return float(num / denom) if denom > 0 else float("nan")


def mean_abs_error(preds, trues):
    return float(np.mean(np.abs(np.asarray(preds, float) - np.asarray(trues, float))))


# ── Model pipelines ───────────────────────────────────────────────────────────

def ols_pipeline():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler",  StandardScaler()),
        ("model",   LinearRegression()),
    ])


def lasso_pipeline():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler",  StandardScaler()),
        ("model",   LassoCV(cv=5, random_state=SEED, max_iter=10_000)),
    ])


def rf_pipeline():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("model",   RandomForestRegressor(
            n_estimators=200, max_features="sqrt", random_state=SEED, n_jobs=-1
        )),
    ])


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading raw data...")
    feat_by_code = {}
    with open(DATA_PATH) as f:
        reader = csv.DictReader(f)
        for row in reader:
            feat_by_code[row["participant.code"]] = get_all_features(row)

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    rows_r   = []  # (trait, N, zs_r,   icl_r,   pc1_r, ols_r, lasso_r, rf_r)
    rows_mae = []  # (trait, N, zs_mae, icl_mae,          ols_mae, lasso_mae, rf_mae)
    lasso_coef_blocks = []

    for target in TRAITS:
        print(f"  {target}...", end=" ", flush=True)
        true_map     = load_true_vals(target)
        icl_pred_map = load_preds(target, "icl")
        zs_pred_map  = load_preds(target, "zero_shot")
        feat_traits  = [t for t in TRAITS if t != target]

        # Build aligned arrays over the ICL participant set
        codes, y_list, X_list = [], [], []
        for code, t_val in true_map.items():
            if code not in feat_by_code:
                continue
            codes.append(code)
            y_list.append(t_val)
            X_list.append([feat_by_code[code].get(ft) for ft in feat_traits])

        N = len(y_list)
        y = np.array(y_list, dtype=float)
        X = np.array(
            [[v if v is not None else np.nan for v in row] for row in X_list],
            dtype=float,
        )

        # ICL / zero-shot performance on the same participant set
        icl_p = [icl_pred_map[c] for c in codes if c in icl_pred_map]
        icl_t = [true_map[c]     for c in codes if c in icl_pred_map]
        zs_p  = [zs_pred_map[c]  for c in codes if c in zs_pred_map]
        zs_t  = [true_map[c]     for c in codes if c in zs_pred_map]

        icl_r_val  = pearson_r(icl_p, icl_t)
        zs_r_val   = pearson_r(zs_p,  zs_t)
        icl_mae_val = mean_abs_error(icl_p, icl_t)
        zs_mae_val  = mean_abs_error(zs_p,  zs_t)

        # PC1 on the same participant set (mirrors pca_baseline.py)
        col_means = np.nanmean(X, axis=0)
        col_stds  = np.nanstd(X, axis=0)
        col_means = np.where(np.isnan(col_means), 0.0, col_means)
        col_stds  = np.where((col_stds == 0) | np.isnan(col_stds), 1.0, col_stds)
        X_z = (X - col_means) / col_stds
        X_z = np.where(np.isnan(X_z), 0.0, X_z)
        _, _, Vt = np.linalg.svd(X_z, full_matrices=False)
        pc1_scores = X_z @ Vt[0]
        pc1_r_val  = abs(pearson_r(pc1_scores.tolist(), y_list))

        # 5-fold CV for all three ML models
        ols_cv   = cross_val_predict(ols_pipeline(),   X, y, cv=kf)
        lasso_cv = cross_val_predict(lasso_pipeline(), X, y, cv=kf)
        rf_cv    = cross_val_predict(rf_pipeline(),    X, y, cv=kf)

        ols_r_val   = pearson_r(ols_cv,   y_list)
        lasso_r_val = pearson_r(lasso_cv, y_list)
        rf_r_val    = pearson_r(rf_cv,    y_list)

        ols_mae_val   = mean_abs_error(ols_cv,   y)
        lasso_mae_val = mean_abs_error(lasso_cv, y)
        rf_mae_val    = mean_abs_error(rf_cv,    y)

        rows_r.append((target, N, zs_r_val, icl_r_val, pc1_r_val,
                       ols_r_val, lasso_r_val, rf_r_val))
        rows_mae.append((target, N, zs_mae_val, icl_mae_val,
                         ols_mae_val, lasso_mae_val, rf_mae_val))

        # Lasso coefficients (fit once on full data for interpretability)
        lasso_full = lasso_pipeline()
        lasso_full.fit(X, y)
        coefs = lasso_full.named_steps["model"].coef_
        alpha_used = lasso_full.named_steps["model"].alpha_
        sorted_coefs = sorted(zip(feat_traits, coefs.tolist()), key=lambda x: -abs(x[1]))
        nonzero = [(ft, c) for ft, c in sorted_coefs if abs(c) > 1e-10]
        block = [f"  {target} (alpha={alpha_used:.5f}, {len(nonzero)}/{len(feat_traits)} features):"]
        if nonzero:
            for ft, c in nonzero:
                block.append(f"    {ft:<28} {c:>+.4f}")
        else:
            block.append("    (all coefficients zeroed — intercept-only model)")
        lasso_coef_blocks.append("\n".join(block))

        print(f"N={N}  OLS={ols_r_val:.3f}  Lasso={lasso_r_val:.3f}  RF={rf_r_val:.3f}")

    # ── Format output ─────────────────────────────────────────────────────────
    W = 96
    lines = [
        "ML Benchmarks vs LLM ICL — BehaviorICL",
        "5-fold CV (OLS / Lasso / RF) vs ICL, Zero-shot, PC1",
        "=" * W,
        "",
        "TABLE 1: Pearson r",
        "",
        (f"{'Trait':<30} {'N':>5}  {'Zero-shot':>10} {'ICL':>8} {'PC1*':>8}"
         f" {'OLS':>8} {'Lasso':>8} {'RF':>8}"),
        "-" * W,
    ]
    for trait, N, zs, icl, pc1, ols, lasso, rf in rows_r:
        lines.append(
            f"{trait:<30} {N:>5}  {zs:>10.4f} {icl:>8.4f} {pc1:>8.4f}"
            f" {ols:>8.4f} {lasso:>8.4f} {rf:>8.4f}"
        )
    lines += [
        "-" * W,
        ("* PC1 uses full dataset (no train/test split); OLS/Lasso/RF use 5-fold CV."
         " ICL/Zero-shot are full-dataset."),
        "",
        "TABLE 2: MAE",
        "",
        (f"{'Trait':<30} {'N':>5}  {'Zero-shot':>10} {'ICL':>8}"
         f" {'OLS':>8} {'Lasso':>8} {'RF':>8}"),
        "-" * W,
    ]
    for trait, N, zs, icl, ols, lasso, rf in rows_mae:
        lines.append(
            f"{trait:<30} {N:>5}  {zs:>10.4f} {icl:>8.4f}"
            f" {ols:>8.4f} {lasso:>8.4f} {rf:>8.4f}"
        )
    lines += [
        "-" * W,
        "",
        "NOTES",
        "  OLS/Lasso/RF: KFold(5, shuffle=True, seed=42). Features: 8 other traits.",
        "  Imputation: mean (within each fold). Scaling: StandardScaler (OLS/Lasso only).",
        "  Lasso: LassoCV with 5-fold inner CV, max_iter=10000.",
        "  RF: 200 trees, max_features='sqrt', random_state=42.",
        "  tolerance_for_inequality true values converted to A-credits (10 − slider).",
        "",
        "LASSO NONZERO COEFFICIENTS (fit on full data; standardized features)",
        "",
    ]
    lines.extend(lasso_coef_blocks)

    output = "\n".join(lines)
    print("\n" + output)

    out_dir = ROOT / "3_Eval" / "cross_trait"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "ml_benchmarks.txt").write_text(output + "\n")
    print(f"\nSaved → {out_dir}/ml_benchmarks.txt")


if __name__ == "__main__":
    main()
