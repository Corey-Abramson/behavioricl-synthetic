"""
PCA Baseline — BehaviorICL

Leave-one-trait-out PC1 benchmark against LLM ICL.

For each held-out trait:
  1. Load the same participants evaluated by ICL for that trait.
  2. Build a feature matrix from the other 8 traits (z-scored, missing → mean-imputed).
  3. Extract PC1 via SVD.
  4. Compute Pearson r (and r²) between PC1 scores and the held-out target.
  5. Compare to LLM ICL r.

Output:
    3_Eval/cross_trait/pca_baseline.txt
"""

import json
import math
import csv
import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "1_Prompts"))
from trait_configs import TRAIT_CONFIGS

TRAITS = [
    "trust", "cooperation", "coordination", "depth_of_reasoning",
    "tolerance_for_inequality", "risk_tolerance", "altruism", "morality",
    "competitiveness",
]

DATA_PATH = ROOT / "0_Preprocess" / "synthetic_data.csv"


# ── Feature extractor: one normalised scalar per participant per trait ─────────

def _val(v):
    try:
        x = float(v)
        return None if math.isnan(x) else x
    except (TypeError, ValueError):
        return None


def get_all_features(row):
    """Return {trait: float or None} for all 9 traits from one data row."""
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
    # slider stores B's credits; corrected A's credits = 10 − stored
    f["tolerance_for_inequality"] = (10 - raw) if raw is not None else None

    f["risk_tolerance"] = _val(row.get("games_solo.1.player.boxes_collected"))
    f["altruism"]        = _val(row.get("games_solo.1.player.dictator_decision"))

    moral = [_val(row.get(f"games_solo.1.player.moral_decision_{n}")) for n in range(1, 6)]
    moral = [v for v in moral if v is not None]
    f["morality"] = sum(moral) if moral else None

    f["competitiveness"] = _val(row.get("compete.1.player.compete_decision"))

    return f


# ── Load ICL true values for a trait ──────────────────────────────────────────

def load_icl_pairs(trait_name):
    """Return {participant_code: true_value} using the same logic as eval_trait.py."""
    config = TRAIT_CONFIGS[trait_name]
    path = ROOT / "2_Inference" / trait_name / "icl_results.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)

    # trust stores keys as true_trust_send / true_trust_return (not dotted path)
    if trait_name == "trust":
        true_keys_direct = ["true_trust_send"]
    else:
        true_keys_direct = None
        true_keys = [f"true_{c.replace('.', '_')}" for c in config.target_cols]

    pairs = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        code = r["participant_code"]
        if true_keys_direct is not None:
            v = r.get(true_keys_direct[0])
            if v is not None:
                pairs[code] = float(v)
        else:
            raw_trues = [r.get(k) for k in true_keys]
            # Skip non-numeric target columns. depth_of_reasoning carries
            # guessing_game.1.player.treatment among its target_cols and that
            # field is a string ("1/3", "1/2", "2/3"); filtering only on
            # `is not None` let it reach sum() and raised TypeError. Same guard
            # as eval_trait.py and ml_benchmarks.py already use.
            valid = []
            for v in raw_trues:
                if v is None:
                    continue
                try:
                    valid.append(float(v))
                except (TypeError, ValueError):
                    pass
            if not valid:
                continue
            t = sum(valid) / len(valid)
            if trait_name == "tolerance_for_inequality":
                t = 10 - t
            pairs[code] = float(t)
    return pairs


# ── Pearson r ─────────────────────────────────────────────────────────────────

def pearson_r(xs, ys):
    n = len(xs)
    if n < 2:
        return float("nan")
    xs, ys = np.array(xs), np.array(ys)
    mx, my = xs.mean(), ys.mean()
    num = ((xs - mx) * (ys - my)).mean()
    den = xs.std() * ys.std()
    return float(num / den) if den > 0 else float("nan")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading raw data...")
    feat_by_code = {}
    with open(DATA_PATH) as f:
        reader = csv.DictReader(f)
        for row in reader:
            feat_by_code[row["participant.code"]] = get_all_features(row)

    header = (
        f"\n{'Trait':<28} {'N':>5}  {'PC1 r':>7}  {'PC1 r²':>7}"
        f"  {'LLM r':>7}  {'LLM r²':>7}  {'Δr (LLM−PC1)':>13}"
    )
    sep = "-" * 78
    rows = []
    loading_blocks = []

    for target in TRAITS:
        icl_pairs = load_icl_pairs(target)  # code → true value
        feature_traits = [t for t in TRAITS if t != target]

        codes, y_vals, X_rows = [], [], []
        for code, t_val in icl_pairs.items():
            if code not in feat_by_code:
                continue
            feat_row = [feat_by_code[code].get(ft) for ft in feature_traits]
            codes.append(code)
            y_vals.append(t_val)
            X_rows.append(feat_row)

        N, M = len(y_vals), len(feature_traits)
        if N == 0:
            print(f"  Skipping {target}: no participants matched")
            continue

        X = np.array([[v if v is not None else np.nan for v in row] for row in X_rows], dtype=float)
        if X.ndim == 1:
            X = X.reshape(N, M)

        # Z-score each column; mean-impute NaNs (→ 0 after z-scoring)
        with np.errstate(all="ignore"):
            col_means = np.nanmean(X, axis=0)
            col_stds  = np.nanstd(X, axis=0)
        col_means = np.where(np.isnan(col_means), 0.0, col_means)
        col_stds  = np.where((col_stds == 0) | np.isnan(col_stds), 1.0, col_stds)
        X_z = (X - col_means) / col_stds
        X_z = np.where(np.isnan(X_z), 0.0, X_z)  # impute with column mean (= 0 after z-score)

        # SVD → PC1
        _, _, Vt = np.linalg.svd(X_z, full_matrices=False)
        pc1_loadings = Vt[0]                    # shape (M,)
        pc1_scores   = X_z @ pc1_loadings       # shape (N,)

        r_pc1 = pearson_r(pc1_scores.tolist(), y_vals)
        # Sign is arbitrary; use abs for the magnitude comparison
        r_pc1_signed = r_pc1
        r_pc1 = abs(r_pc1)

        # LLM ICL r — computed directly from inference files on the same participant set
        config = TRAIT_CONFIGS[target]
        pred_key = "pred_trust_send" if target == "trust" else f"pred_{config.prediction_key}"
        icl_path = ROOT / "2_Inference" / target / "icl_results.jsonl"
        icl_map = {r["participant_code"]: r for r in
                   (json.loads(l) for l in icl_path.read_text().splitlines() if l.strip())}
        llm_preds, llm_trues = [], []
        for code, t_val in zip(codes, y_vals):
            rec = icl_map.get(code)
            if rec is None:
                continue
            p = rec.get(pred_key)
            if p is not None:
                llm_preds.append(float(p))
                llm_trues.append(t_val)
        r_llm = pearson_r(llm_preds, llm_trues)

        r2_pc1 = r_pc1 ** 2
        r2_llm = r_llm ** 2 if not math.isnan(r_llm) else float("nan")
        delta   = r_llm - r_pc1

        rows.append((target, N, r_pc1_signed, r_pc1, r2_pc1, r_llm, r2_llm, delta))

        # PC1 loading block (sorted by |loading|)
        sorted_loads = sorted(zip(feature_traits, pc1_loadings.tolist()), key=lambda x: -abs(x[1]))
        block = [f"  {target}:"]
        for ft, ld in sorted_loads:
            block.append(f"    {ft:<28} {ld:>+.4f}")
        loading_blocks.append("\n".join(block))

    # ── Print table ──────────────────────────────────────────────────────────
    lines = [
        "PCA Baseline vs LLM ICL — Leave-One-Trait-Out",
        "=" * 78,
        header,
        sep,
    ]
    for target, N, r_signed, r_pc1, r2_pc1, r_llm, r2_llm, delta in rows:
        sign_note = " (−)" if r_signed < 0 else ""
        lines.append(
            f"{target:<28} {N:>5}  {r_pc1:>7.4f}{sign_note:<4}"
            f"  {r2_pc1:>7.4f}  {r_llm:>7.4f}  {r2_llm:>7.4f}  {delta:>+13.4f}"
        )
    lines.append(sep)
    lines.append(
        "PC1 r = |Pearson r| between leave-one-out PC1 score and held-out trait.\n"
        "(−) = PC1 anti-correlated with target (sign is arbitrary; magnitude used for comparison).\n"
        "Δr = LLM ICL r − PC1 |r|."
    )
    lines.append("\n\nPC1 loadings (sorted by |loading|):")
    lines.extend(loading_blocks)

    output = "\n".join(lines)
    print(output)

    out_dir = ROOT / "3_Eval" / "cross_trait"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "pca_baseline.txt").write_text(output + "\n")
    print(f"\nSaved → {out_dir}/pca_baseline.txt")


if __name__ == "__main__":
    main()
