"""One figure for the top of the five-year-gate scoring section.

Failure at the five-year proof of continued use by within-class-and-cohort
quintile of lead and of atypicality, under the three theme constructions
(50 global themes, 500 global themes, 50 themes fitted per class), from
paper/results/resolution_compare.json. Per-quintile binomial SE bars.

Output: paper/results/fig_resolution_compare.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "paper" / "results"

LABELS = {"global50": "50 global themes (production)",
          "global500": "500 global themes",
          "perclass50": "50 themes fitted per class"}
COLORS = {"global50": "#222222", "global500": "#2b6cb0", "perclass50": "#c0392b"}
STYLES = {"global50": "-", "global500": "--", "perclass50": ":"}


def main() -> int:
    j = json.loads((RES / "resolution_compare.json").read_text())
    pooled = j["pooled"]
    scorings = [k for k in ("global50", "global500", "perclass50") if k in pooled]
    xs = [1, 2, 3, 4, 5]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), sharey=True)
    for ax, var, lab in ((axes[0], "L", "lead"), (axes[1], "A", "atypicality")):
        for s in scorings:
            rec = pooled[s][var]
            n5 = rec["n"] / 5
            p = rec["quintiles"]
            se = [(v * (1 - v) / n5) ** 0.5 for v in p]
            ax.errorbar(xs, [100 * v for v in p], yerr=[196 * v for v in se],
                        fmt="o" + STYLES[s], color=COLORS[s], lw=1.8, ms=4,
                        capsize=2, label=LABELS[s])
        ax.set_xlabel(f"quintile of {lab}, within Nice class and registration year")
        ax.set_xticks(xs)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("% failing the five-year proof of continued use")
    axes[0].legend(fontsize=8, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(RES / "fig_resolution_compare.png", dpi=150, bbox_inches="tight")
    print("[done] fig_resolution_compare.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
