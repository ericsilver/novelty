"""One figure: the quintile profile of every outcome, on a common axis.

The paper states in three separate places that a linear-in-position summary
misrepresents the data -- registration is an inverse-U in lead, first-gate
failure is convex, and listing is U-shaped in lead -- and each time the reader
has to reconstruct the shape from numbers in prose. This draws them.

Everything plotted is already estimated elsewhere; nothing is re-fit here.
Panels A and B are within-cell linear probability coefficients relative to the
omitted bottom quintile, re-expressed as adjusted rates: the sample base rate
plus each quintile's coefficient less the mean coefficient, so the five bars
average to the base rate and read in percent. The intervals are the reported
standard errors of the coefficients, which are intervals on differences from
the bottom quintile, not on the levels. Panel C is raw binned rates, because
the registration profile is reported that way in the source table.

No parametric curve is fitted through five points. The claim is that the
profile is not a line, not that it is any particular other function.

Panels A and B are coefficients relative to the omitted bottom quintile, so
they are not on a common vertical scale with panel C, which is a raw rate. The
three panels also come from three different samples and cell definitions --
registrations within class x cohort, registered debut owners within class x
year, and all debut filings -- which is why the figure is three panels rather
than one.

Because everything is read from JSON, the figure inherits whatever scoring
those JSONs were produced under. Regenerate them with the same SURPRISE_SRC
before regenerating this, or the panels will disagree with each other.

Inputs : paper/results/gate_decisive_regression.json   (spec 2_FE_controls)
         paper/results/debut_edgar_substantiate.json   (specs A2, B3)
         paper/results/debut_outcome_topic.json        (raw registration rates)
Output : paper/results/quintile_profiles.{png,pdf}
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

INK = "#1a1a1a"
INK2 = "#555555"
GRID = "#d8d8d8"
C_LEAD = "#c0392b"     # the signed axis
C_ATYP = "#2b6cb0"     # the unsigned axis
QX = [1, 2, 3, 4, 5]


def load(name: str) -> dict:
    return json.loads((RES / name).read_text())


def spec(doc: dict, label: str) -> dict:
    for s in doc["specs"]:
        if s["label"] == label:
            return s
    raise KeyError(label)


def profile(sp: dict, prefix: str) -> tuple[list[float], list[float]]:
    """Coefficients relative to Q1 (which is the omitted category, hence 0)."""
    b = [0.0]
    se = [0.0]
    for q in range(2, 6):
        c = sp["coef"][f"{prefix}{q}"]
        b.append(c["b"] * 100)
        se.append(c["se"] * 100)
    return b, se


def band(ax, x, b, se, color, label, marker="o"):
    lo = [bi - 1.96 * si for bi, si in zip(b, se)]
    hi = [bi + 1.96 * si for bi, si in zip(b, se)]
    ax.fill_between(x, lo, hi, color=color, alpha=0.13, linewidth=0)
    ax.plot(x, b, marker + "-", color=color, lw=2.0, ms=5.5, label=label)


def style(ax, title, ylab):
    ax.set_xticks(QX)
    ax.set_xticklabels(["Q1\nlow", "Q2", "Q3", "Q4", "Q5\nhigh"], fontsize=8.5)
    ax.set_title(title, fontsize=10, color=INK, pad=8)
    ax.set_ylabel(ylab, fontsize=9, color=INK2)
    ax.grid(alpha=0.28, color=GRID)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8.5)


def main() -> int:
    gate = load("gate_decisive_regression.json")
    edgar = load("debut_edgar_substantiate.json")
    debut = load("debut_outcome_topic.json")

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.3))

    # ---- A: first-gate failure, on lead ------------------------------------
    sp = spec(gate, "2_FE_controls")
    b, se = profile(sp, "q")
    base = sp["p_fail1"] * 100
    adj = [base + bi - sum(b) / 5 for bi in b]
    ax = axes[0]
    ax.bar(QX, adj, width=0.62, color=C_LEAD, alpha=0.85, zorder=2)
    ax.errorbar(QX, adj, yerr=[1.96 * v for v in se], fmt="none", ecolor=INK,
                elinewidth=0.9, capsize=3, zorder=3)
    ax.axhline(base, color=INK2, lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax.text(0.55, base, f"all registrations {base:.1f}%", fontsize=7.4,
            color=INK2, va="bottom", ha="left")
    lo = min(adj) - 1.2; hi = max(adj) + 1.2
    ax.set_ylim(lo, hi)
    for x, v in zip(QX, adj):
        ax.text(x, v + 0.12, f"{v:.1f}", ha="center", va="bottom", fontsize=7.6, color=INK)
    step = adj[4] - adj[3]
    ax.annotate(f"+{step:.1f} from Q4 to Q5,\nseven times the average step below",
                xy=(4.5, (adj[3] + adj[4]) / 2), xytext=(1.0, hi - 0.25),
                fontsize=7.4, color=INK2, ha="left", va="top",
                arrowprops=dict(arrowstyle="->", color=INK2, lw=0.9,
                                connectionstyle="arc3,rad=-0.2"))
    style(ax, "A. Failing the first use-proof gate\n(within Nice class and registration year, controls)",
          "% failing the gate, adjusted")
    ax.set_xlabel("lead $L$ quintile", fontsize=9, color=INK2)

    # ---- B: listing, on each axis ------------------------------------------
    ax = axes[1]
    spa, spd = spec(edgar, "A2_atyp_len"), spec(edgar, "B3_dKL_len_levels")
    ba, sea = profile(spa, "aq")
    bd, sed = profile(spd, "dq")
    base_b = spa["p_sec"] * 100
    adja = [base_b + v - sum(ba) / 5 for v in ba]
    adjd = [base_b + v - sum(bd) / 5 for v in bd]
    w = 0.36
    ax.bar([x - w / 2 for x in QX], adja, width=w, color=C_ATYP, alpha=0.85,
           label="by atypicality $A$", zorder=2)
    ax.bar([x + w / 2 for x in QX], adjd, width=w, color=C_LEAD, alpha=0.85,
           label="by lead $L$ (levels held)", zorder=2)
    ax.errorbar([x - w / 2 for x in QX], adja, yerr=[1.96 * v for v in sea], fmt="none",
                ecolor=INK, elinewidth=0.8, capsize=2.5, zorder=3)
    ax.errorbar([x + w / 2 for x in QX], adjd, yerr=[1.96 * v for v in sed], fmt="none",
                ecolor=INK, elinewidth=0.8, capsize=2.5, zorder=3)
    ax.axhline(base_b, color=INK2, lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax.text(0.55, base_b, f"all registered debuts {base_b:.2f}%", fontsize=7.4,
            color=INK2, va="bottom", ha="left")
    allv = adja + adjd
    ax.set_ylim(min(allv) - 0.08, max(allv) + 0.08)
    style(ax, "B. Reaching the listed universe\n(within Nice class and filing year, length held)",
          "% of owners ever SEC-reporting, adjusted")
    ax.set_xlabel("quintile of the axis shown", fontsize=9, color=INK2)
    ax.legend(fontsize=7.8, loc="lower left", framealpha=0.9)

    # ---- C: registration, twenty bins -------------------------------------
    # Five bins turn a non-monotone profile into a sawtooth. Twenty show the
    # real shape: an inverse-U on atypicality, and on lead a trough just above
    # the median that is compositional -- |lead| correlates with atypicality at
    # +0.31, so the middle of the lead distribution is disproportionately
    # low-atypicality filings, which register poorly.
    ax = axes[2]
    shape = load("registration_shape.json")
    xs = [(i + 0.5) / 20 * 100 for i in range(20)]
    ax.plot(xs, shape["bins"]["atyp"], "-", color=C_ATYP, lw=2.0,
            label="on atypicality $A$")
    ax.plot(xs, shape["bins"]["lead"], "-", color=C_LEAD, lw=2.0, label="on lead $L$")
    mid = shape["lead_within_atyp_quintile"][2]
    ax.plot(xs, mid, "--", color=C_LEAD, lw=1.4, alpha=0.75,
            label="on lead, middle $A$ quintile")
    ax.axhline(debut["p_registered"] * 100, color=INK2, lw=1.0, ls=(0, (4, 3)))
    ax.set_title("C. Completing registration\n(raw rates, twenty bins)", fontsize=10,
                 color=INK, pad=8)
    ax.set_ylabel("% reaching registration", fontsize=9, color=INK2)
    ax.set_xlabel("percentile of the axis shown", fontsize=9, color=INK2)
    ax.grid(alpha=0.28, color=GRID)
    for sp_ in ax.spines.values():
        sp_.set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8.5)
    ax.legend(fontsize=7.4, loc="lower right", framealpha=0.9)
    tr = min(range(20), key=lambda i: shape["bins"]["lead"][i])
    ax.annotate("trough is composition:\nlow-$|L|$ filings are\nlow-$A$ filings",
                xy=(xs[tr], shape["bins"]["lead"][tr]), xytext=(6, 46.5),
                fontsize=7.4, color=INK2, ha="left",
                arrowprops=dict(arrowstyle="->", color=INK2, lw=0.9,
                                connectionstyle="arc3,rad=-0.25"))

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(RES / f"quintile_profiles.{ext}", dpi=150, bbox_inches="tight")
    print("[done] quintile_profiles.{png,pdf}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
