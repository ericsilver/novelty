"""Figure: the within-firm dKL-patent relation does not differ by IPR-complementarity.

Reads paper/results/patent_complementarity_by_sector.json (written by
scripts/patent_complementarity_by_sector.py) and draws a coefficient plot of the
pooled estimate against the complement / neutral / substitute sector groups, under
both scorings, with 95% confidence intervals.

The referee prediction being tested (Castaldi, 2026-07-24): "I would expect
correlation in specific sectors where the two IPRs are complementary." Under the
paper's primary topic scoring the complement-vs-substitute slope gap is zero.

Outputs: paper/results/patent_complementarity.{png,pdf}
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "paper" / "results"

# Categorical slots 1 and 2 of the validated reference palette. Documented as
# clearing every adjacent- and all-pairs gate in both modes for the first three
# slots, so this pair needs no re-validation.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

ROWS = [
    ("pooled", "All firms\n(pooled)"),
    ("complement", "Complement sectors\npharma, chemicals, machinery,\nelectronics, devices, sci/tech"),
    ("neutral", "Neutral sectors"),
    ("substitute", "Substitute sectors\napparel, food, retail, finance,\ntransport, hospitality"),
]

SERIES = [
    ("t200", "Topic scoring (T=200) — the paper's primary measure", BLUE, "o"),
    ("token", "Term scoring", ORANGE, "s"),
]


def cell(scoring: dict, key: str) -> tuple[float, float] | None:
    """Return (coef, se) for a row key under one scoring, or None if absent."""
    if key == "pooled":
        node = scoring.get("pooled")
    else:
        node = (scoring.get("by_ipr_group") or {}).get(key)
    if not node or "logpat" not in node:
        return None
    lp = node["logpat"]
    return float(lp["coef"]), float(lp["se"])


def main() -> None:
    data = json.loads((RES / "patent_complementarity_by_sector.json").read_text())
    scorings = data["scorings"]

    fig, ax = plt.subplots(figsize=(9.2, 5.4))

    # Row y-positions, top row first.
    y_of = {k: len(ROWS) - 1 - i for i, (k, _) in enumerate(ROWS)}
    offsets = {"t200": +0.15, "token": -0.15}

    ax.axvline(0, color=INK, lw=1.2, zorder=2)
    ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)

    for skey, _label, color, marker in SERIES:
        sc = scorings.get(skey, {})
        for rkey, _ in ROWS:
            got = cell(sc, rkey)
            if got is None:
                continue
            coef, se = got
            y = y_of[rkey] + offsets[skey]
            lo, hi = coef - 1.96 * se, coef + 1.96 * se
            ax.plot([lo, hi], [y, y], color=color, lw=2.0,
                    solid_capstyle="round", zorder=3)
            ax.plot([coef], [y], marker=marker, ms=9, color=color,
                    markeredgecolor="#fcfcfb", markeredgewidth=1.6, zorder=4)
            ax.annotate(f"{coef:+.3f}", (coef, y), textcoords="offset points",
                        xytext=(0, 11 if skey == "t200" else -17),
                        ha="center", fontsize=8.5, color=INK_2)

    ax.set_yticks(list(y_of.values()))
    ax.set_yticklabels([lab for _, lab in ROWS], fontsize=9, color=INK)
    ax.set_ylim(-0.6, len(ROWS) - 0.4)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", colors=MUTED, labelsize=9)

    ax.set_xlabel("Within-firm coefficient on log(1 + patents), in standard deviations of "
                  "$\\Delta$KL\n(firm and year fixed effects, standard errors clustered on firm; "
                  "bars are 95% intervals)",
                  fontsize=9, color=INK_2)

    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)

    ax.set_title("Trademark vocabulary position is unrelated to patenting within firm,\n"
                 "in complement and substitute sectors alike",
                 fontsize=12, color=INK, loc="left", pad=26)

    inter = (scorings.get("t200", {}).get("interaction_complement") or {}).get("logpat_x_comp")
    if inter:
        ax.annotate("Complement $\\times$ log(patents) interaction under topic scoring: "
                    f"{float(inter['coef']):+.4f} (t = {float(inter['t']):.2f})",
                    xy=(0.0, 1.02), xycoords="axes fraction",
                    fontsize=9.5, color=INK_2, ha="left", va="bottom")

    handles = [Line2D([0], [0], color=c, marker=m, ms=8, lw=2.0,
                      markeredgecolor="#fcfcfb", markeredgewidth=1.4, label=lab)
               for _k, lab, c, m in SERIES]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.26),
              ncol=2, frameon=False, fontsize=9, labelcolor=INK_2,
              handletextpad=0.6, columnspacing=2.4)

    n_obs = scorings.get("t200", {}).get("pooled", {}).get("n_obs")
    n_firms = scorings.get("t200", {}).get("pooled", {}).get("n_firms")
    if n_obs and n_firms:
        ax.annotate(f"{n_obs:,} firm-years, {n_firms:,} firms with at least three years of "
                    f"trademark filings matched to PatentsView assignees, 1995–2019.\n"
                    f"Every firm in the panel patents at least once, which biases the design "
                    f"toward finding complementarity.",
                    xy=(0.0, -0.40), xycoords="axes fraction",
                    fontsize=8.5, color=MUTED, ha="left", va="top")

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(RES / f"patent_complementarity.{ext}", dpi=200, bbox_inches="tight",
                    facecolor="#fcfcfb")
    print(f"[done] -> {RES / 'patent_complementarity.png'}")


if __name__ == "__main__":
    main()
