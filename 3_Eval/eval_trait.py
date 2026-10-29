"""
Trait Evaluation — BehaviorICL

Evaluates all three methods (ICL, zero-shot, other-student ICL) for a given trait.
Reads results from 2_Inference/{trait}/
Writes to 3_Eval/{trait}/

Usage:
    python eval_trait.py <trait_name>
    python eval_trait.py competitiveness

Metrics:
    MAE, Pearson r, bootstrap CIs (B=10,000), Wilcoxon signed-rank (ICL vs zero-shot),
    SD calibration, exact match, within-tolerance accuracy

Outputs:
    3_Eval/{trait}/summary.txt
    3_Eval/{trait}/validation_stats.txt
"""

import json
import math
import random
import sys
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "1_Prompts"))

from trait_configs import TRAIT_CONFIGS, TraitConfig

B = 10_000
SEED = 42


# ── Stat helpers ──────────────────────────────────────────────────────────────

def pearson_r(xs, ys):
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((xs[i] - mx) * (ys[i] - my) for i in range(n)) / n
    sdx = math.sqrt(sum((x - mx) ** 2 for x in xs) / n)
    sdy = math.sqrt(sum((y - my) ** 2 for y in ys) / n)
    return cov / (sdx * sdy) if sdx > 0 and sdy > 0 else float("nan")


MIN_FINITE_SHARE = 0.95


def _percentile_ci(draws, b):
    """95% interval over the finite draws; NaN bounds if fewer than 95% are finite.

    A resample whose predictions (or truths) are all equal has an undefined r, and
    NaN cannot be sorted, so undefined draws are dropped before taking percentiles.
    """
    finite = sorted(d for d in draws if math.isfinite(d))
    m = len(finite)
    if m < MIN_FINITE_SHARE * b:
        return float("nan"), float("nan")
    return finite[int(0.025 * m)], finite[int(0.975 * m)]


def bootstrap_r(pred, true, b=B, seed=SEED):
    rng = random.Random(seed)
    n = len(pred)
    rs = []
    for _ in range(b):
        idx = [rng.randint(0, n - 1) for _ in range(n)]
        rs.append(pearson_r([pred[i] for i in idx], [true[i] for i in idx]))
    return _percentile_ci(rs, b)


def bootstrap_r_diff(pred_a, pred_b, true, b=B, seed=SEED):
    rng = random.Random(seed)
    n = len(true)
    diffs = []
    for _ in range(b):
        idx = [rng.randint(0, n - 1) for _ in range(n)]
        ra = pearson_r([pred_a[i] for i in idx], [true[i] for i in idx])
        rb = pearson_r([pred_b[i] for i in idx], [true[i] for i in idx])
        diffs.append(ra - rb)
    return _percentile_ci(diffs, b)


def wilcoxon_signed_rank(diffs):
    nonzero = [(d, i) for i, d in enumerate(diffs) if d != 0]
    n = len(nonzero)
    if n == 0:
        return 0, 1.0, 0.0
    ranked = sorted(nonzero, key=lambda x: abs(x[0]))
    ranks = []
    i = 0
    while i < n:
        j = i
        while j < n and abs(ranked[j][0]) == abs(ranked[i][0]):
            j += 1
        avg_rank = (i + 1 + j) / 2
        for k in range(i, j):
            ranks.append((ranked[k][0], avg_rank))
        i = j
    W_plus = sum(r for d, r in ranks if d > 0)
    W_minus = sum(r for d, r in ranks if d < 0)
    W = min(W_plus, W_minus)
    mean_w = n * (n + 1) / 4
    std_w = math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    z = (W - mean_w) / std_w
    p = 2 * (1 - _norm_cdf(abs(z)))
    rb = 1 - (2 * W) / (n * (n + 1) / 2 + 1e-9)
    return W, p, rb


def _norm_cdf(z):
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def sd(vals):
    n = len(vals)
    m = sum(vals) / n
    return math.sqrt(sum((x - m) ** 2 for x in vals) / n)


def mae(pairs):
    return sum(abs(p - t) for p, t in pairs) / len(pairs)


def exact(pairs):
    return sum(1 for p, t in pairs if p == t) / len(pairs)


def within(pairs, tol):
    return sum(1 for p, t in pairs if abs(p - t) <= tol) / len(pairs)


# ── Compute true value for a trait from a result record ───────────────────────

def get_pred_true(records: list[dict], config: TraitConfig) -> tuple[list[float], list[float]]:
    """
    Returns (preds, trues) for the primary target column.

    For multi-column targets (cooperation rounds 1-4, coordination rounds 1-4),
    we use the *mean* across columns as the single comparison value, matching the
    prediction_range definition in TraitConfig. Morality (decisions 1-5) is asked
    as a count of donations, so its true value is the *sum* of the five decisions;
    a participant missing any of the five is skipped.
    """
    # DIVERGENCE FROM THE ORIGINAL PIPELINE (synthetic export only).
    # run_trait.py writes trust results under pred_trust_send / pred_trust_return
    # and true_trust_send / true_trust_return, not under pred_<prediction_key>.
    # The original eval_trait.py therefore finds no predictions for trust at all
    # and dies with ZeroDivisionError; the trust numbers in the paper came from a
    # separate two-output evaluator that is not in the repository. The branch
    # below scores trust on trust_send, which is the same primary outcome
    # pca_baseline.py and ml_benchmarks.py already use for this trait.
    # A fuller fix would score both outputs.
    if config.name == "trust":
        pred_key = "pred_trust_send"
        true_keys = ["true_trust_send"]
    else:
        pred_key = f"pred_{config.prediction_key}"
        true_keys = [f"true_{col.replace('.', '_')}" for col in config.target_cols]

    preds, trues = [], []
    for r in records:
        p = r.get(pred_key)
        if p is None:
            continue
        # Aggregate multi-column true value (skip None and non-numeric metadata fields)
        raw_trues = [r.get(k) for k in true_keys]
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
        if config.name == "morality":
            # prediction is a count, 0 to 5: score against the count
            if len(valid) < len(true_keys):
                continue
            t = sum(valid)
        else:
            t = sum(valid) / len(valid)
        # redistribute_decision stores B's credits (slider position); predictions
        # are for A's credits, so convert: true_A = 10 - stored_B
        if config.name == "tolerance_for_inequality":
            t = 10 - t
        preds.append(float(p))
        trues.append(float(t))

    return preds, trues


# ── Within-tolerance by trait ──────────────────────────────────────────────────

WITHIN_TOLERANCE = {
    "trust": 0.5,
    "competitiveness": 0.0,       # binary: exact match only
    "cooperation": 0.1,
    "coordination": 0.25,
    "depth_of_reasoning": 5.0,
    "tolerance_for_inequality": 1.0,
    "risk_tolerance": 1.0,
    "altruism": 1.0,
    "morality": 0.5,
}


# ── Load results ──────────────────────────────────────────────────────────────

def load_results(trait_name: str) -> tuple[list, list, list]:
    """Returns (icl_records, zs_records, oth_records) aligned by participant_code."""
    base = ROOT / "2_Inference" / trait_name

    def load(path):
        if not path.exists():
            raise FileNotFoundError(f"Results not found: {path}")
        return {r["participant_code"]: r for r in
                (json.loads(l) for l in path.read_text().splitlines() if l.strip())}

    icl_map = load(base / "icl_results.jsonl")
    zs_map  = load(base / "zero_shot_results.jsonl")
    oth_map = load(base / "other_student_icl_results.jsonl")

    codes = sorted(set(icl_map) & set(zs_map) & set(oth_map))
    return (
        [icl_map[c] for c in codes],
        [zs_map[c]  for c in codes],
        [oth_map[c] for c in codes],
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main(trait_name: str):
    if trait_name not in TRAIT_CONFIGS:
        print(f"Unknown trait: {trait_name}")
        print(f"Available: {list(TRAIT_CONFIGS.keys())}")
        sys.exit(1)

    config = TRAIT_CONFIGS[trait_name]
    icl_recs, zs_recs, oth_recs = load_results(trait_name)
    N = len(icl_recs)
    print(f"Participants in all three: {N}")

    out_dir = Path(__file__).parent / trait_name
    out_dir.mkdir(exist_ok=True)

    # Extract aligned preds and trues
    icl_pred, true = get_pred_true(icl_recs, config)
    zs_pred,  _    = get_pred_true(zs_recs,  config)
    oth_pred, _    = get_pred_true(oth_recs,  config)

    tol = WITHIN_TOLERANCE.get(trait_name, 1.0)
    pairs_icl = list(zip(icl_pred, true))
    pairs_zs  = list(zip(zs_pred,  true))
    pairs_oth = list(zip(oth_pred, true))

    # ── Summary table ──────────────────────────────────────────────────────────
    header = f"{'Method':<22} {'MAE':>8} {'Exact':>8} {f'±{tol}':>8} {'Pearson r':>10}"
    sep = "-" * len(header)
    lines = [f"BehaviorICL Evaluation — {config.label}  (N={N})", sep, header, sep]

    for label, pairs, pred in [
        ("ICL",              pairs_icl, icl_pred),
        ("Zero-shot",        pairs_zs,  zs_pred),
        ("Other-student ICL",pairs_oth, oth_pred),
    ]:
        m = mae(pairs)
        e = exact(pairs)
        w = within(pairs, tol)
        r = pearson_r(pred, true)
        lines.append(f"{label:<22} {m:>8.4f} {e:>8.4f} {w:>8.4f} {r:>10.4f}")

    lines.append(sep)
    summary = "\n".join(lines)
    print("\n" + summary)
    (out_dir / "summary.txt").write_text(summary + "\n")

    # ── Statistical validation ─────────────────────────────────────────────────
    vlines = [f"Statistical Validation — {config.label}  N={N}", "=" * 60]

    # Bootstrap CIs
    vlines.append("\n## 1. Bootstrap CIs on Pearson r (B=10,000)\n")
    for label, pred in [("ICL", icl_pred), ("Zero-shot", zs_pred), ("Other-student ICL", oth_pred)]:
        r = pearson_r(pred, true)
        lo, hi = bootstrap_r(pred, true)
        vlines.append(f"  {label:<22} r={r:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")

    vlines.append("")
    diff = pearson_r(icl_pred, true) - pearson_r(zs_pred, true)
    lo_d, hi_d = bootstrap_r_diff(icl_pred, zs_pred, true)
    vlines.append(f"  ICL − Zero-shot:  Δr={diff:.4f}  95% CI [{lo_d:.4f}, {hi_d:.4f}]")

    # Wilcoxon
    vlines.append("\n## 2. Wilcoxon Signed-Rank: ICL vs Zero-shot (absolute error)\n")
    icl_err = [abs(icl_pred[i] - true[i]) for i in range(len(true))]
    zs_err  = [abs(zs_pred[i]  - true[i]) for i in range(len(true))]
    diffs   = [icl_err[i] - zs_err[i] for i in range(len(true))]
    W, p, rb = wilcoxon_signed_rank(diffs)
    direction = "ICL better" if sum(1 for d in diffs if d < 0) > sum(1 for d in diffs if d > 0) else "Zero-shot better"
    vlines.append(f"  W={W:.0f}, p={p:.2e}, rank-biserial r={rb:.4f} ({direction})")
    vlines.append(f"  Mean |error| — ICL: {sum(icl_err)/len(icl_err):.4f}  Zero-shot: {sum(zs_err)/len(zs_err):.4f}")

    # SD calibration
    vlines.append("\n## 3. SD Calibration\n")
    true_sd = sd(true)
    header_sd = f"  {'Method':<24} {'SD':>6}   {'% of true SD':>13}"
    vlines.append(header_sd)
    vlines.append("  " + "-" * (len(header_sd) - 2))
    for label, pred in [("True", true), ("ICL", icl_pred), ("Zero-shot", zs_pred), ("Other-student ICL", oth_pred)]:
        s = sd(pred)
        pct = s / true_sd * 100
        vlines.append(f"  {label:<24} {s:>6.4f}   {pct:>12.1f}%")

    vout = "\n".join(vlines)
    print("\n" + vout)
    (out_dir / "validation_stats.txt").write_text(vout + "\n")

    print(f"\nSaved → {out_dir}/summary.txt")
    print(f"Saved → {out_dir}/validation_stats.txt")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python eval_trait.py <trait_name>")
        print(f"Available traits: {list(TRAIT_CONFIGS.keys())}")
        sys.exit(1)
    main(sys.argv[1])
