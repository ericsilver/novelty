"""Appendix figure: why the construct is scored on topic distributions, not raw terms.

Argument shown rather than asserted. Under TERM scoring a filing's atypicality
A = (KL_past + KL_future)/2 and its document LENGTH are very nearly the same
variable, so the lead L = KL_past - KL_future inherits a variance that grows
with position along the diagonal. Under TOPIC scoring (T=200) the two come
apart.

Each cell is a hexbin over filings in (KL_past, KL_future), every hexagon
coloured by the MEAN FILING LENGTH (n_terms) of the filings inside it, on a
shared log colour scale across all four panels.

Outputs
    paper/results/representation_appendix.png
    paper/results/representation_appendix.pdf
    paper/results/representation_appendix_corr.csv     corr(A, log n_terms)
    paper/results/representation_appendix_sdl.csv      sd(L) by A-decile
    paper/results/representation_appendix_exemplars.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
RES = REPO / "paper" / "results"

# ---------------------------------------------------------------- conventions
INK = "#0b0b0b"
SECOND = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BG = "#ffffff"
DPI = 170

# Sequential, single hue, light -> dark. Length has no meaningful midpoint, so
# nothing diverging; the ramp starts at a step that is still visible against a
# light background rather than at the background colour itself.
LENGTH_CMAP = LinearSegmentedColormap.from_list(
    "length_seq",
    ["#e3ece9", "#a9cbc7", "#63a6a3", "#2f7a7c", "#14535c", "#08303a"],
)

# Hexagons holding fewer than this many filings are left blank, so that no
# hexagon's colour is a mean over a handful of filings. At gridsize 55 this
# floor still retains >99.4% of the filings in every panel.
MINCNT = 25
GRIDSIZE = 55

# Shared colour limits across all four panels, so that the same colour means
# the same mean length everywhere and "flatter gradient" is directly visible.
VMIN, VMAX = 5.0, 600.0

# The paper's scored window. Outside it the reference on one side is burn-in:
# the 1990 cohort in particular has a past reference of a few hundred filings,
# and Dirichlet smoothing then lifts KL-against-the-past by a near-constant
# amount for every filing that year, which drew a second band parallel to the
# diagonal in an earlier version of this figure.
YEAR_LO, YEAR_HI = 1995, 2019
CLEAN = (
    (pl.col("n_ref_past") >= 1000)
    & (pl.col("n_ref_future") >= 1000)
    & (pl.col("n_terms") >= 3)
    & pl.col("year").is_between(YEAR_LO, YEAR_HI)
)

# Exemplars are keyed on (mark text, filing year, owner substring) -- all three
# are required, because mark text plus year is NOT unique in this corpus and
# matching on two alone has plotted the wrong company here before. Class 009
# has an unrelated IPOD (2001) filed by Newport Financial Corporation; class
# 035 has AMAZON BOOKS (1997), which is the filing actually owned by
# "Amazon.com, Inc." -- the AMAZON.COM filing itself sits under the assignee of
# record, AMAZON TECHNOLOGIES, INC. The resolved serial is asserted against the
# expected serial so that a silent re-resolution cannot slip through.
ROWS = [
    ("009", "Nice class 009 — Software & Electronics",
     ("IPOD", 2001, "APPLE", 75982871, "IPOD (2001, Apple)")),
    ("035", "Nice class 035 — Advertising & Retail",
     ("AMAZON.COM", 1997, "AMAZON TECHNOLOGIES", 75277670,
      "AMAZON.COM (1997, Amazon)")),
]
COLS = [("token", "Term scoring"), ("topic", "Topic scoring (T = 200)")]


def load(cls: str, scoring: str) -> pl.DataFrame:
    """Metadata and the clean filter come from the class-year token file, the
    only one carrying mark_identification / owner_name. The coordinates come
    from the per-filing-anchored files -- the production anchoring -- for both
    representations: termroll_* for terms, rolling_*_T200 for topics. Both
    write topic_-prefixed column names."""
    sp = pl.read_parquet(PROC / f"surprise_class{cls}.parquet").filter(CLEAN)
    src = (PROC / f"termroll_surprise_class{cls}.parquet" if scoring == "token"
           else PROC / f"rolling_surprise_class{cls}_T200.parquet")
    tp = pl.read_parquet(
        src, columns=["serial_number", "topic_kl_vs_past", "topic_kl_vs_future"],
    ).filter(pl.col("topic_kl_vs_past").is_finite()
             & pl.col("topic_kl_vs_future").is_finite())
    sp = sp.drop("kl_vs_past", "kl_vs_future").join(
        tp, on="serial_number", how="inner"
    ).rename({"topic_kl_vs_past": "kl_vs_past",
              "topic_kl_vs_future": "kl_vs_future"})
    return sp.with_columns([
        ((pl.col("kl_vs_past") + pl.col("kl_vs_future")) / 2).alias("A"),
        (pl.col("kl_vs_past") - pl.col("kl_vs_future")).alias("L"),
    ])


def resolve_exemplar(df: pl.DataFrame, mark: str, year: int, owner: str,
                     expect_serial: int) -> dict:
    """All three of mark / year / owner are required, and the serial is then
    asserted. Mark text is matched as a whole token so that AMAZON.COM does not
    also catch AMAZON BOOKS; the dot is escaped so it is not a wildcard."""
    esc = mark.replace(".", r"\.")
    hits = df.filter(
        (pl.col("year") == year)
        & pl.col("mark_identification").str.to_uppercase().str.contains(
            f"^{esc}$|^{esc} ")
        & pl.col("owner_name").str.to_uppercase().str.contains(owner)
    ).sort(pl.col("kl_vs_past") - pl.col("kl_vs_future"), descending=True)
    if hits.is_empty():
        raise SystemExit(f"exemplar not found: {mark} ({year}) owner~{owner}")
    r = hits.row(0, named=True)
    if int(r["serial_number"]) != expect_serial:
        raise SystemExit(
            f"exemplar resolved to serial {r['serial_number']}, expected "
            f"{expect_serial} -- refusing to plot a possibly wrong filing")
    return r


def decile_table(d: pl.DataFrame) -> list[dict]:
    a = d["A"].to_numpy()
    L = d["L"].to_numpy()
    nt = d["n_terms"].to_numpy()
    edges = np.percentile(a, np.arange(0, 101, 10))
    idx = np.clip(np.searchsorted(edges, a, side="right") - 1, 0, 9)
    out = []
    for i in range(10):
        m = idx == i
        out.append({"decile": i + 1, "n": int(m.sum()),
                    "A_mean": float(a[m].mean()),
                    "sd_L": float(L[m].std(ddof=1)),
                    "median_n_terms": float(np.median(nt[m]))})
    return out


def main() -> int:
    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": BG,
        "text.color": INK, "axes.labelcolor": SECOND,
        "xtick.color": SECOND, "ytick.color": SECOND,
        "axes.edgecolor": GRID, "font.size": 9,
    })
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 11.0))
    corr_rows, sdl_rows, exemplar_rows = [], [], []

    for i, (cls, row_label, ex_key) in enumerate(ROWS):
        for j, (scoring, col_label) in enumerate(COLS):
            ax = axes[i][j]
            d = load(cls, scoring)
            x = d["kl_vs_past"].to_numpy()
            y = d["kl_vs_future"].to_numpy()
            length = d["n_terms"].to_numpy().astype(float)

            # Each panel's box comes from its own data: token KL sits on a
            # common 2-10 scale but topic KL does not, so raw magnitudes are
            # not comparable across columns.
            both = np.concatenate([x, y])
            lo = float(np.percentile(both, 0.5))
            hi = float(np.percentile(both, 99.5))

            r_A_logn = float(np.corrcoef(d["A"].to_numpy(), np.log(length))[0, 1])
            r_levels = float(np.corrcoef(x, y)[0, 1])
            tab = decile_table(d)
            sds = np.array([t["sd_L"] for t in tab])
            spread = float(sds.max() / sds.min())
            corr_rows.append({
                "nice_class": cls, "scoring": scoring, "n": d.height,
                "corr_A_log_nterms": round(r_A_logn, 4),
                "corr_kl_past_future": round(r_levels, 4),
                "sd_L_min": round(float(sds.min()), 4),
                "sd_L_max": round(float(sds.max()), 4),
                "sd_L_spread": round(spread, 3),
                "median_nterms_decile1": tab[0]["median_n_terms"],
                "median_nterms_decile10": tab[9]["median_n_terms"],
                "box_lo": round(lo, 3), "box_hi": round(hi, 3),
            })
            for t in tab:
                sdl_rows.append({"nice_class": cls, "scoring": scoring, **t})

            ax.set_axisbelow(True)
            ax.grid(color=GRID, linewidth=0.6, alpha=0.9)
            hb = ax.hexbin(x, y, C=length, reduce_C_function=np.mean,
                           gridsize=GRIDSIZE, extent=(lo, hi, lo, hi),
                           mincnt=MINCNT, cmap=LENGTH_CMAP,
                           norm=LogNorm(vmin=VMIN, vmax=VMAX),
                           linewidths=0.0, zorder=2)
            ax.plot([lo, hi], [lo, hi], color=INK, linewidth=0.8,
                    linestyle=(0, (5, 4)), zorder=3)
            ax.text(0.975, 0.955, "45$^\\circ$", transform=ax.transAxes,
                    fontsize=7.5, ha="right", va="top",
                    color=SECOND, zorder=4,
                    bbox=dict(boxstyle="round,pad=0.15", fc=BG, ec="none",
                              alpha=0.85))

            # Exemplar: same serial in both columns, so the reader watches one
            # filing move between representations. Marked by a ringed marker
            # AND a text label -- never colour alone.
            mark, year, owner, expect, disp = ex_key
            r = resolve_exemplar(d, mark, year, owner, expect)
            ex, ey = float(r["kl_vs_past"]), float(r["kl_vs_future"])
            ax.scatter([ex], [ey], s=95, marker="o", facecolor="none",
                       edgecolor=BG, linewidth=2.6, zorder=5)
            ax.scatter([ex], [ey], s=95, marker="o", facecolor="none",
                       edgecolor=INK, linewidth=1.3, zorder=6)
            ax.scatter([ex], [ey], s=7, marker="o", color=INK, zorder=6)
            exemplar_rows.append({
                "nice_class": cls, "scoring": scoring,
                "serial_number": int(r["serial_number"]),
                "mark_identification": r["mark_identification"],
                "filing_date": r["filing_date"], "owner_name": r["owner_name"],
                "n_terms": int(r["n_terms"]),
                "kl_vs_past": round(ex, 4), "kl_vs_future": round(ey, 4),
                "A": round((ex + ey) / 2, 4), "L": round(ex - ey, 4),
            })

            # Label placed inside the box, below-right of the marker, then
            # nudged back if it would escape the axes.
            span = hi - lo
            lx, ly = ex + 0.045 * span, ey - 0.075 * span
            ha = "left"
            if lx + 0.34 * span > hi:
                lx, ha = ex - 0.045 * span, "right"
            ly = max(ly, lo + 0.04 * span)
            ax.annotate(
                f"{disp}\n$n$ = {int(r['n_terms'])} terms",
                xy=(ex, ey), xytext=(lx, ly), fontsize=8.2, color=INK,
                ha=ha, va="top", zorder=7, linespacing=1.35,
                bbox=dict(boxstyle="round,pad=0.28", fc=BG, ec=GRID,
                          linewidth=0.8, alpha=0.94),
                arrowprops=dict(arrowstyle="-", color=SECOND, linewidth=0.7,
                                shrinkA=1, shrinkB=7))

            # The headline number, printed in the panel it belongs to.
            ax.text(0.035, 0.965,
                    f"$r$(A, log length) = {r_A_logn:+.3f}\n"
                    f"sd($L$) across A-deciles: {sds.min():.3f}–{sds.max():.3f}"
                    f"  ({spread:.2f}×)",
                    transform=ax.transAxes, ha="left", va="top", fontsize=8.4,
                    color=INK, linespacing=1.5, zorder=7,
                    bbox=dict(boxstyle="round,pad=0.36", fc=BG, ec=GRID,
                              linewidth=0.9, alpha=0.95))

            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.tick_params(length=3, width=0.6)
            for s in ax.spines.values():
                s.set_linewidth(0.8)
            unit = "term-scored nats" if scoring == "token" else "topic-scored nats"
            ax.set_xlabel(f"KL against the past  ({unit})", fontsize=8.6)
            ax.set_ylabel(f"KL against the future  ({unit})", fontsize=8.6)
            if i == 0:
                ax.set_title(col_label, fontsize=11.5, color=INK, pad=26,
                             fontweight="semibold")
            ax.text(0.5, 1.012, f"$n$ = {d.height:,} filings",
                    transform=ax.transAxes, ha="center", va="bottom",
                    fontsize=8.2, color=MUTED)

        # Row label, set outside the left panel's y-axis.
        axes[i][0].text(-0.20, 0.5, row_label, transform=axes[i][0].transAxes,
                        rotation=90, ha="right", va="center", fontsize=11,
                        color=INK, fontweight="semibold")

    fig.subplots_adjust(left=0.115, right=0.975, top=0.915, bottom=0.185,
                        wspace=0.26, hspace=0.30)
    cax = fig.add_axes([0.30, 0.112, 0.42, 0.015])
    cb = fig.colorbar(hb, cax=cax, orientation="horizontal", extend="both")
    cb.set_ticks([5, 10, 25, 50, 100, 250, 600])
    cb.ax.set_xticklabels(["5", "10", "25", "50", "100", "250", "600"])
    cb.ax.tick_params(labelsize=8, length=2, width=0.6, color=SECOND, pad=2)
    cb.outline.set_linewidth(0.5)
    cb.outline.set_edgecolor(GRID)
    cb.set_label("mean filing length within hexagon (terms, log scale)",
                 fontsize=8.6, color=SECOND, labelpad=5)

    fig.text(0.5, 0.056,
             "Hexagons holding fewer than "
             f"{MINCNT} filings are left blank; the retained hexagons still "
             "cover over 99.4% of filings in every panel.",
             ha="center", va="top", fontsize=7.9, color=MUTED)
    fig.text(0.5, 0.035,
             "The colour scale is shared by all four panels. The axis ranges "
             "are not: each panel is boxed on its own 0.5th–99.5th "
             "percentiles, and term-scored and",
             ha="center", va="top", fontsize=7.9, color=MUTED)
    fig.text(0.5, 0.014,
             "topic-scored KL do not share a scale — the comparison invited "
             "across columns is of shape and colour gradient, not of raw "
             "magnitude.",
             ha="center", va="top", fontsize=7.9, color=MUTED)

    RES.mkdir(parents=True, exist_ok=True)
    fig.savefig(RES / "representation_appendix.png", dpi=DPI,
                facecolor=BG)
    fig.savefig(RES / "representation_appendix.pdf", facecolor=BG)
    plt.close(fig)

    # Composition control. Topic scoring can only score about two thirds of the
    # filings (the rest have no finite topic KL), so part of the flattening in
    # the right-hand column could in principle be a different sample rather
    # than a different representation. Re-running TERM scoring on exactly the
    # filings topic scoring keeps settles it.
    for cls, _row_label, _ex in ROWS:
        tok = load(cls, "token")
        top = load(cls, "topic").select("serial_number")
        kept = tok.join(top, on="serial_number", how="semi")
        a = kept["A"].to_numpy()
        r_A_logn = float(np.corrcoef(a, np.log(kept["n_terms"].to_numpy()))[0, 1])
        tab = decile_table(kept)
        sds = np.array([t["sd_L"] for t in tab])
        corr_rows.append({
            "nice_class": cls, "scoring": "token_on_topic_subset",
            "n": kept.height,
            "corr_A_log_nterms": round(r_A_logn, 4),
            "corr_kl_past_future": round(float(np.corrcoef(
                kept["kl_vs_past"], kept["kl_vs_future"])[0, 1]), 4),
            "sd_L_min": round(float(sds.min()), 4),
            "sd_L_max": round(float(sds.max()), 4),
            "sd_L_spread": round(float(sds.max() / sds.min()), 3),
            "median_nterms_decile1": tab[0]["median_n_terms"],
            "median_nterms_decile10": tab[9]["median_n_terms"],
            "box_lo": None, "box_hi": None,
        })
        for t in tab:
            sdl_rows.append({"nice_class": cls,
                             "scoring": "token_on_topic_subset", **t})

    pl.from_dicts(corr_rows).write_csv(RES / "representation_appendix_corr.csv")
    pl.from_dicts(sdl_rows).write_csv(RES / "representation_appendix_sdl.csv")
    pl.from_dicts(exemplar_rows).write_csv(
        RES / "representation_appendix_exemplars.csv")

    for r in corr_rows:
        print(f"[{r['nice_class']}/{r['scoring']}] n={r['n']:>9,} "
              f"r(A,log len)={r['corr_A_log_nterms']:+.3f} "
              f"r(past,fut)={r['corr_kl_past_future']:.3f} "
              f"sd(L) {r['sd_L_min']:.3f}-{r['sd_L_max']:.3f} "
              f"({r['sd_L_spread']:.2f}x) "
              f"med len d1={r['median_nterms_decile1']:.0f} "
              f"d10={r['median_nterms_decile10']:.0f}", file=sys.stderr)
    for r in exemplar_rows:
        print(f"  exemplar {r['nice_class']}/{r['scoring']}: "
              f"{r['mark_identification']} serial={r['serial_number']} "
              f"{r['filing_date']} {r['owner_name']} n_terms={r['n_terms']} "
              f"A={r['A']:.3f} L={r['L']:+.3f}", file=sys.stderr)
    print("[done] paper/results/representation_appendix.png", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
