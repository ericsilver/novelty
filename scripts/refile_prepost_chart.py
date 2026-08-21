"""Appendix chart: what a refiled description looks like before and after the rewrite.

Reads paper/results/refile_text_change.json (prepost.rewritten_only) and draws
two panels of paired points -- atypicality and description length, before and
after -- for refilings by self-filers, by counsel, and for the four
representation transitions. Rewritten pairs only; identical-text refilings
move on neither axis.

Output: paper/results/refile_prepost.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "paper" / "results"
INK, INK2, GRID = "#1a1a1a", "#555555", "#d8d8d8"
PRE, POST = "#9aa5b1", "#c0392b"

ORDER = [("self-filed refiling", "Self-filed refiling"),
         ("counsel refiling", "Counsel refiling"),
         (None, None),
         ("self->self", "Self, then self"),
         ("self->counsel", "Self, then counsel"),
         ("counsel->counsel", "Counsel, then counsel"),
         ("counsel->self", "Counsel, then self")]


def main() -> int:
    d = json.loads((RES / "refile_text_change.json").read_text())["prepost"]["rewritten_only"]
    rows = [(k, lab) for k, lab in ORDER]
    ys = list(range(len(rows)))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, key, title, xlab, fmt in (
            (axes[0], "A", "Atypicality", "mean atypicality (nats)", "{:.2f}"),
            (axes[1], "len", "Description length", "mean words in goods/services text", "{:.0f}")):
        for y, (k, lab) in zip(ys, rows):
            if k is None:
                continue
            r = d[k]
            a, b = r[f"{key}_pre"], r[f"{key}_post"]
            ax.plot([a, b], [y, y], color=INK2, lw=1.4, zorder=1)
            ax.scatter([a], [y], color=PRE, s=46, zorder=2, edgecolor=INK, lw=0.5)
            ax.scatter([b], [y], color=POST, s=46, zorder=3, edgecolor=INK, lw=0.5)
            ax.annotate(fmt.format(a), (a, y), xytext=(0, 7), textcoords="offset points",
                        ha="center", fontsize=7.4, color=INK2)
            ax.annotate(fmt.format(b), (b, y), xytext=(0, -12), textcoords="offset points",
                        ha="center", fontsize=7.4, color=POST)
        ax.set_yticks(ys)
        ax.set_yticklabels([f"{lab}  (n = {d[k]['n']:,})" if k else "" for k, lab in rows],
                           fontsize=8.6)
        ax.set_title(title, fontsize=10, color=INK, pad=8)
        ax.set_xlabel(xlab, fontsize=9, color=INK2)
        ax.grid(axis="x", alpha=0.3, color=GRID)
        for s in ax.spines.values():
            s.set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=8.5)
    axes[0].scatter([], [], color=PRE, edgecolor=INK, lw=0.5, s=46, label="abandoned filing")
    axes[0].scatter([], [], color=POST, edgecolor=INK, lw=0.5, s=46, label="registered refiling")
    axes[0].legend(fontsize=8, loc="lower left", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(RES / "refile_prepost.png", dpi=160, bbox_inches="tight")
    print("[done] refile_prepost.png", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
