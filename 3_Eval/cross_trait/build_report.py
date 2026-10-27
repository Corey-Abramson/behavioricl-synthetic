"""
Cross-Trait Aggregate Report — BehaviorICL

Reads summary.txt and validation_stats.txt for all 9 traits,
assembles a unified report and heatmap.

Outputs:
  cross_trait_report.txt   — aggregate table + narrative
  heatmap.png              — trait × method Pearson r heatmap
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
EVAL_DIR = Path(__file__).parent.parent

TRAITS = [
    "trust",
    "competitiveness",
    "cooperation",
    "coordination",
    "depth_of_reasoning",
    "tolerance_for_inequality",
    "risk_tolerance",
    "altruism",
    "morality",
]

LABELS = {
    "trust":                   "Trust",
    "competitiveness":         "Competitiveness",
    "cooperation":             "Cooperation",
    "coordination":            "Coordination",
    "depth_of_reasoning":      "Depth of Reasoning",
    "tolerance_for_inequality":"Tolerance for Inequality",
    "risk_tolerance":          "Risk Tolerance",
    "altruism":                "Altruism",
    "morality":                "Morality",
}

# ── Parsers ───────────────────────────────────────────────────────────────────

def parse_summary(trait: str) -> dict:
    path = EVAL_DIR / trait / "summary.txt"
    text = path.read_text()
    rows = {}
    for line in text.splitlines():
        for method in ["ICL", "Zero-shot", "Other-student ICL"]:
            if line.strip().startswith(method):
                nums = re.findall(r"[-\d]+\.[\d]+", line)
                if len(nums) >= 4:
                    rows[method] = {
                        "mae": float(nums[0]),
                        "exact": float(nums[1]),
                        "within": float(nums[2]),
                        "r": float(nums[3]),
                    }
    return rows


def parse_validation(trait: str) -> dict:
    path = EVAL_DIR / trait / "validation_stats.txt"
    text = path.read_text()
    result = {}

    # Bootstrap CIs
    for method_key, pattern in [
        ("icl",  r"ICL\s+r=([-\d.]+)\s+95% CI \[([-\d.]+),\s*([-\d.]+)\]"),
        ("zs",   r"Zero-shot\s+r=([-\d.]+)\s+95% CI \[([-\d.]+),\s*([-\d.]+)\]"),
        ("oth",  r"Other-student ICL\s+r=([-\d.]+)\s+95% CI \[([-\d.]+),\s*([-\d.]+)\]"),
    ]:
        m = re.search(pattern, text)
        if m:
            result[f"{method_key}_r"] = float(m.group(1))
            result[f"{method_key}_ci_lo"] = float(m.group(2))
            result[f"{method_key}_ci_hi"] = float(m.group(3))

    # ICL - Zero-shot delta (generic form or trust-specific "(send r)" form)
    m = re.search(r"ICL − Zero-shot(?:\s*\([^)]*\))?:\s+Δr=([-\d.]+)\s+95% CI \[([-\d.]+),\s*([-\d.]+)\]", text)
    if m:
        result["delta_r"] = float(m.group(1))
        result["delta_lo"] = float(m.group(2))
        result["delta_hi"] = float(m.group(3))

    # Wilcoxon
    m = re.search(r"W=([\d.]+), p=([\d.e+-]+), rank-biserial r=([-\d.]+) \(([^)]+)\)", text)
    if m:
        result["wilcoxon_p"] = float(m.group(2))
        result["wilcoxon_rb"] = float(m.group(3))
        result["wilcoxon_dir"] = m.group(4)

    # SD calibration
    m = re.search(r"ICL\s+([\d.]+)\s+([\d.]+)%", text)
    if m:
        result["icl_sd_pct"] = float(m.group(2))

    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    out_dir = Path(__file__).parent
    out_dir.mkdir(exist_ok=True)

    # Gather all data
    data = {}
    for t in TRAITS:
        try:
            data[t] = {
                "summary": parse_summary(t),
                "validation": parse_validation(t),
            }
        except FileNotFoundError as e:
            print(f"Warning: missing data for {t}: {e}")

    # ── Table 1: Cross-trait Pearson r by method ──────────────────────────────
    lines = [
        "BehaviorICL — Cross-Trait Results",
        "=" * 80,
        "",
        "## Table 1: Pearson r by Trait and Method",
        "",
        f"{'Trait':<28} {'N':>6}  {'ICL r':>8}  {'95% CI':>16}  {'ZS r':>7}  {'Δr':>7}  {'CI incl. 0?':>12}",
        "-" * 90,
    ]

    # sort by ICL r descending
    sorted_traits = sorted(
        [t for t in TRAITS if t in data],
        key=lambda t: data[t]["validation"].get("icl_r", -99),
        reverse=True,
    )

    for t in sorted_traits:
        v = data[t]["validation"]
        s = data[t]["summary"]
        icl_r  = v.get("icl_r", float("nan"))
        ci_lo  = v.get("icl_ci_lo", float("nan"))
        ci_hi  = v.get("icl_ci_hi", float("nan"))
        zs_r   = v.get("zs_r", float("nan"))
        delta  = v.get("delta_r", float("nan"))
        dlo    = v.get("delta_lo", float("nan"))
        dhi    = v.get("delta_hi", float("nan"))
        incl0  = "yes" if dlo <= 0 <= dhi else "NO"

        # get N from summary
        n_str = ""
        for method_data in s.values():
            break
        try:
            txt = (EVAL_DIR / t / "summary.txt").read_text()
            m = re.search(r"N=(\d+)", txt)
            n_str = m.group(1) if m else "?"
        except Exception:
            n_str = "?"

        lines.append(
            f"{LABELS[t]:<28} {n_str:>6}  {icl_r:>8.4f}  [{ci_lo:.3f}, {ci_hi:.3f}]  "
            f"{zs_r:>7.4f}  {delta:>7.4f}  {incl0:>12}"
        )

    lines += [
        "-" * 90,
        "CI = bootstrap 95% CI on ICL Pearson r (B=10,000)",
        "Δr = ICL r − Zero-shot r; 'CI incl. 0?' = whether Δr CI crosses zero",
        "",
    ]

    # ── Table 2: MAE comparison ───────────────────────────────────────────────
    lines += [
        "## Table 2: MAE by Trait and Method",
        "",
        f"{'Trait':<28}  {'ICL MAE':>9}  {'ZS MAE':>9}  {'Oth MAE':>9}  {'ICL wins?':>10}",
        "-" * 72,
    ]

    for t in sorted_traits:
        s = data[t]["summary"]
        icl_mae = s.get("ICL", {}).get("mae", float("nan"))
        zs_mae  = s.get("Zero-shot", {}).get("mae", float("nan"))
        oth_mae = s.get("Other-student ICL", {}).get("mae", float("nan"))
        wins = "YES" if icl_mae < zs_mae else "no"
        lines.append(
            f"{LABELS[t]:<28}  {icl_mae:>9.4f}  {zs_mae:>9.4f}  {oth_mae:>9.4f}  {wins:>10}"
        )

    lines += ["-" * 72, ""]

    # ── Table 3: Wilcoxon results ─────────────────────────────────────────────
    lines += [
        "## Table 3: Wilcoxon Signed-Rank (ICL vs Zero-shot absolute error)",
        "",
        f"{'Trait':<28}  {'p-value':>10}  {'rank-bis r':>11}  {'Direction':>20}",
        "-" * 76,
    ]

    for t in sorted_traits:
        v = data[t]["validation"]
        p   = v.get("wilcoxon_p", float("nan"))
        rb  = v.get("wilcoxon_rb", float("nan"))
        dir_ = v.get("wilcoxon_dir", "?")
        sig = "**" if p < 0.05 else "  "
        lines.append(f"{LABELS[t]:<28}  {p:>10.2e}  {rb:>11.4f}  {sig}{dir_:<20}")

    lines += ["-" * 76, "** p < 0.05", ""]

    # ── Table 4: SD calibration ───────────────────────────────────────────────
    lines += [
        "## Table 4: SD Calibration (ICL predicted SD as % of true SD)",
        "",
        f"{'Trait':<28}  {'ICL SD %':>10}",
        "-" * 42,
    ]
    for t in sorted_traits:
        v = data[t]["validation"]
        pct = v.get("icl_sd_pct", float("nan"))
        lines.append(f"{LABELS[t]:<28}  {pct:>9.1f}%")
    lines += ["-" * 42, ""]

    # ── Narrative summary ─────────────────────────────────────────────────────
    # REMOVED IN THE SYNTHETIC EXPORT.
    # The original version of this function appended a hand-written summary
    # paragraph with the study's own Pearson r values typed into the string
    # literals. Those are results computed from the original participant data,
    # so they were stripped from this export along with the data itself. The
    # table above is computed from whatever is in 3_Eval/<trait>/ and is the
    # only thing this script now reports.
    lines += [
        "## Summary",
        "",
        "No narrative summary in this export: the original one hard-coded the",
        "study's own correlation coefficients, which are not shipped here.",
        "The table above reflects the run that produced 3_Eval/<trait>/.",
    ]

    report = "\n".join(lines)
    print(report)

    out_path = out_dir / "cross_trait_report.txt"
    out_path.write_text(report + "\n")
    print(f"\nSaved → {out_path}")

    # ── Heatmap ───────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        import numpy as np

        trait_labels = [LABELS[t] for t in sorted_traits]
        methods = ["ICL", "Zero-shot", "Other-student ICL"]
        method_keys = ["icl_r", "zs_r", "oth_r"]

        matrix = []
        for t in sorted_traits:
            v = data[t]["validation"]
            row = [v.get(k, float("nan")) for k in method_keys]
            matrix.append(row)

        matrix = np.array(matrix)

        fig, ax = plt.subplots(figsize=(7, 8))
        vmax = max(0.4, float(np.nanmax(np.abs(matrix))))
        im = ax.imshow(matrix, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")

        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, fontsize=11)
        ax.set_yticks(range(len(trait_labels)))
        ax.set_yticklabels(trait_labels, fontsize=11)

        # Annotate cells
        for i in range(len(sorted_traits)):
            for j in range(len(methods)):
                val = matrix[i, j]
                if not np.isnan(val):
                    color = "black" if abs(val) < vmax * 0.6 else "white"
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                            fontsize=9, color=color, fontweight="bold")

        plt.colorbar(im, ax=ax, label="Pearson r", shrink=0.6)
        ax.set_title("Trait Predictability by Method\n(Pearson r, ICL vs Baselines)",
                     fontsize=13, fontweight="bold", pad=12)
        plt.tight_layout()

        heatmap_path = out_dir / "heatmap.png"
        plt.savefig(heatmap_path, dpi=150, bbox_inches="tight")
        print(f"Heatmap → {heatmap_path}")

    except ImportError:
        print("matplotlib not installed — skipping heatmap. Run: pip install matplotlib numpy")


if __name__ == "__main__":
    main()
