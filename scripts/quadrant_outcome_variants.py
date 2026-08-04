"""Three outcome-encoded variants of the quadrant scatter (Figure 1).

The published quadrant figure (scripts/quadrant_regen.py -> paper/results/quadrant.png)
shows WHERE a filing sits in vocabulary space but says nothing about what happened
to it.  This script produces three variants on the same axes and the same two class
panels, each adding one outcome dimension.

VARIANT A -- survival at the first Section 8 use-proof gate.
    paper/results/quadrant_survival.{png,pdf}
    A registration FAILS the first gate if case_events.parquet carries a `C8..`
    event at registration age 4.0-8.5 years, which is the construction used by
    scripts/event_gates_all.py and scripts/staged_outcomes_table.py::gate_outcome.
    Only registrations whose gate window has fully elapsed can be classified.
    The C8 event stream ends 2026-04, so the last fully-elapsed registration
    cohort is 2017 (2017-12 + 8.5y = 2026-06 would already be censored; 2002-2017
    is used, one year tighter than event_gates_all.py's 2002-2018, which is
    slightly censored at its top end).
    Everything else -- never registered, or registered too recently -- is NOT
    classifiable and is never drawn as a survivor.  It is shown as a separate,
    labelled marker class in the exemplar layer and is excluded from the surface.

VARIANT B -- filings whose owner reached the public-company universe.
    paper/results/quadrant_public.{png,pdf}
    Density/contour treatment of the same restricted sample (see PUBLIC below).

VARIANT C -- SEC-listing rate surface.
    paper/results/quadrant_listing.{png,pdf}
    Same axes and the same hexbin treatment as variant A, but each cell is
    coloured by the share of filings in it whose owner reached the listed
    universe, expressed as log2 of the ratio to that panel's own base rate.
    A ratio scale (rather than a raw share) is used because the base rates are
    small and differ between panels (4.8% in class 009, 2.5% in class 032):
    on a raw-share scale the whole surface would sit in the bottom few percent
    of the colour range and the two panels would not be comparable.  The log
    of the ratio is used rather than the ratio itself because the ratio is
    bounded below by 0 and unbounded above, so a linear diverging scale centred
    on 1 would compress every depleted cell into [0,1] while stretching every
    enriched one; log2 makes "half the base rate" and "twice the base rate"
    equal distances from the centre.

PUBLIC.  "Public" is CIK-keyed, never owner-name-keyed: a CIK counts if it
    appears in data/processed/sec_firm_year.parquet (the SEC Financial Statement
    Data Sets = the XBRL filer universe) or carries an EDGAR 8-A12B/8-A12G
    registration (`in_8a` in funding_owner_match.parquet = the actual moment of
    exchange listing).  Owner strings are then attached to those CIKs through
    BOTH name->CIK tables (funding_owner_match.parquet and
    uspto_sec_crosswalk.parquet) and matched on the shared normalized key `norm`,
    not on the raw string, so a firm flagged under one USPTO spelling is flagged
    under all of them.
    This deliberately does NOT use `in_sec` (name-keyed, internally inconsistent
    across 1,912 CIKs) and does NOT use `in_fsds`/`in_8a` membership alone:
    every row of funding_owner_match is a Reg D (Form D) match, so those markers
    silently restrict the "public" universe to firms that also filed a Form D
    notice in 2009q1-2026q2.  Under that literal restriction class 032 loses
    Coca-Cola and PepsiCo and class 009 loses Apple and IBM.  Both definitions
    are counted and reported; the wider one is the one plotted.

COMPLETENESS GUARD (corpus edge).  The retrospective axis needs a five-year
    forward reference window, but the token scorer computes kl_vs_future
    whenever ANY forward filings exist.  The corpus runs to 2026-04, so the
    forward window Y+1..Y+5 is complete only for filing years Y <= 2020;
    2021-2025 filings are scored against 4/5, 3/5, ... of a window.  See
    forward_coverage() for the volume-weighted coverage fraction.  Consequences:
      - Variant A's surface is unaffected: its sample is registrations 2002-2017,
        whose filing years top out at 2017, so it contains zero short-window
        filings.
      - Variant C's surface WOULD be affected: filings after 2020 are 24% of the
        in-box sample in both panels.  Its surface is therefore capped to filing
        years 1995-2020 (post-burn-in, complete forward window).
      - Exemplars are never dropped for this reason; short-window exemplars are
        drawn with a distinct ring and a dagger on the label.

EXEMPLARS.  Keyed on (mark text, filing year, required owner substring), exactly
    as in scripts/quadrant_regen.py.  Mark text plus year is NOT unique and
    matching on those two alone silently plotted the wrong company for several
    points.  Do not add an entry here without an owner key.

Overplotting.  Both panels carry 10^5-10^6 eligible filings and the outcome
groups occupy almost exactly the same region, so a two-colour scatter of raw
points is mush at any alpha.  Variants A and C therefore draw a hexbin RATE
SURFACE (cells below a minimum count left blank) plus a small foreground layer
of named exemplars.  Variant B scatters the restricted sample itself, since it
is small enough to show.

Outputs: paper/results/quadrant_survival.{png,pdf}
         paper/results/quadrant_public.{png,pdf}
         paper/results/quadrant_listing.{png,pdf}
         paper/results/quadrant_outcome_counts.csv
         paper/results/quadrant_outcome_exemplars.csv
         paper/results/quadrant_surface_diagnostics.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from adjustText import adjust_text

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
RES = REPO / "paper" / "results"
sys.path.insert(0, str(REPO / "scripts"))
from sec_link import normalize  # noqa: E402

# ---- palette (validated categorical slots 1 and 2, plus repo neutrals) -------
BLUE = "#2a78d6"
ORANGE = "#eb6834"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
PAPER = "#faf9f5"
DIVERGING = LinearSegmentedColormap.from_list(
    "kl_div", [BLUE, "#9dc0ea", "#eeece5", "#f4b193", ORANGE])
# Colour semantics are held constant across the surfaces: BLUE always means the
# better commercial outcome and ORANGE the worse one. Variant A encodes the
# FAILURE rate, so high values are bad and DIVERGING is used as-is. Variant C
# encodes the LISTING rate, where high values are good, so it uses the reversed
# ramp -- otherwise the same colour would mean "survived" on one figure and
# "never listed" on the other.
DIVERGING_R = LinearSegmentedColormap.from_list(
    "kl_div_r", [ORANGE, "#f4b193", "#eeece5", "#9dc0ea", BLUE])

# (mark, filing year, required owner substring).  The owner is part of the key
# because mark text plus year is NOT unique -- see module docstring and
# scripts/quadrant_regen.py, which this list mirrors exactly.
PANELS = [
    ("009", "Software & Electronics", [
        ("OPENAI", 2016, "OPENAI"), ("CLAUDE", 2023, "ANTHROPIC"),
        ("INSTAGRAM", 2011, "INSTAGRAM"), ("ETHEREUM", 2018, "ETHEREUM"),
        ("KUBERNETES", 2014, "LF PROJECTS"), ("CHATGPT", 2022, "OPENAI"),
        ("UBER", 2010, "UBER TECHNOLOGIES"), ("PHOTOSHOP", 2003, "ADOBE"),
        ("AIRPODS", 2015, "APPLE"), ("KINDLE", 2010, "AMAZON"),
        ("IPOD", 2001, "APPLE"),
    ]),
    ("032", "Beer & Soft Drinks", [
        ("LIQUID DEATH", 2023, "SUPPLYING DEMAND"),
        ("WHITE CLAW SELTZER WORKS", 2016, "MARK ANTHONY"),
        ("ROCKSTAR", 2002, "ROCKSTAR"), ("HARD MTN DEW", 2021, "PEPSICO"),
        ("FIJI", 2005, "FIJI WATER"), ("SMARTWATER", 2000, "ENERGY BRANDS"),
        ("KOMBUCHA", 1997, ""), ("POWERADE", 1996, "COCA-COLA"),
        ("GATORADE", 1996, "STOKELY"), ("RED BULL", 1995, "RED BULL GMBH"),
        ("MONSTER ENERGY", 2002, "MONSTER"), ("CELSIUS", 2004, "ELITE FX"),
    ]),
]

CLEAN = (
    (pl.col("n_ref_past") >= 1000)
    & (pl.col("n_ref_future") >= 1000)
    & (pl.col("n_terms") >= 3)
)
LO, HI = 2.0, 10.0
GATE_LO, GATE_HI = 4.0, 8.5      # first Section 8 window, registration-age years
REG_LO, REG_HI = 2002, 2017      # cohorts whose first gate has fully elapsed
FWD_YEARS = 5                    # length of the forward reference window
COVERAGE_MIN = 0.95              # a filing needs ~all five forward years
SURF_LO, SURF_HI = 1995, 2020    # variant C surface: post burn-in, full window

# (gridsize, min points per hex).  Both raised to ~3x the cell count of the
# original (26, 250) / (14, 120) settings -- gridsize x sqrt(3), mincnt / 3.
HEX_SURV = {"009": (45, 85), "032": (24, 40)}
HEX_LIST = {"009": (45, 400), "032": (18, 250)}
SURV_PP = 16.0                   # colour range, pp either side of panel base
LIST_L2 = 2.0                    # colour range, log2 units either side of 1x
Z_MARK = 8                       # exemplar markers sit ABOVE their own labels


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# ----------------------------------------------------------------- data -----
def c8_first() -> pl.DataFrame:
    return (
        pl.scan_parquet(PROC / "case_events.parquet")
        .filter((pl.col("code") == "C8..") & (pl.col("date") > 19000000))
        .select("serial_number", "date").collect()
        .with_columns(pl.col("date").cast(pl.Utf8)
                      .str.strptime(pl.Date, "%Y%m%d", strict=False).alias("c8_d"))
        .drop_nulls("c8_d").group_by("serial_number").agg(pl.col("c8_d").min())
    )


def public_keys() -> tuple[set[str], set[str], set[str]]:
    """(wide public norm keys, 8-A-only norm keys, literal in_fsds|in_8a names)."""
    fsds = set(pl.read_parquet(PROC / "sec_firm_year.parquet",
                               columns=["cik"])["cik"].to_list())
    fund = pl.read_parquet(PROC / "funding_owner_match.parquet")
    xw = pl.read_parquet(PROC / "uspto_sec_crosswalk.parquet")
    ciks_8a = set(fund.filter(pl.col("in_8a") == 1)["cik"].to_list())
    pub_ciks = fsds | ciks_8a
    wide = set(fund.filter(pl.col("cik").is_in(list(pub_ciks)))["norm"].to_list())
    wide |= set(xw.filter(pl.col("cik").is_in(list(pub_ciks)))["norm"].to_list())
    only8a = set(fund.filter(pl.col("in_8a") == 1)["norm"].to_list())
    only8a |= set(xw.filter(pl.col("cik").is_in(list(ciks_8a)))["norm"].to_list())
    literal = set(fund.filter((pl.col("in_fsds") == 1) | (pl.col("in_8a") == 1))
                  ["owner_name"].to_list())
    log(f"[public] {len(fsds):,} FSDS CIKs + {len(ciks_8a):,} 8-A CIKs "
        f"-> {len(wide):,} normalized owner keys ({len(only8a):,} listed via 8-A); "
        f"literal in_fsds|in_8a owner strings: {len(literal):,}")
    return wide, only8a, literal


def forward_coverage(cls: str) -> dict[int, float]:
    """Share of each filing year's five-year forward window the corpus covers.

    A filing in year Y is scored against filings in Y+1..Y+5.  The corpus ends
    partway through 2026, so recent years are scored against a truncated window.
    Coverage is weighted by filing volume: observed filings in the window over
    the filings the window would hold if the years past the corpus edge ran at
    the last complete year's rate.  This is deliberately NOT a comparison of
    n_ref_future against a global median -- the corpus grows by an order of
    magnitude over the sample, so a global median flags 1990s filings with
    complete windows (RED BULL 1995 sits at 0.33 of the class-032 median) while
    missing genuinely truncated ones (CHATGPT 2022 sits at 0.83 of the class-009
    median).  Volume-weighted calendar coverage separates the two.
    """
    n = (pl.scan_parquet(PROC / f"surprise_class{cls}.parquet")
         .group_by("year").agg(pl.len().alias("n")).collect().sort("year"))
    counts = {int(r["year"]): int(r["n"]) for r in n.iter_rows(named=True)}
    last_full = max(y for y in counts if counts[y] >= 0.5 * counts[max(counts)])
    # last calendar year believed complete = the last year before the partial tail
    ymax = max(counts)
    last_full = ymax - 1 if counts[ymax] < 0.5 * counts[ymax - 1] else ymax
    rate = counts[last_full]
    out = {}
    for y in sorted(counts):
        obs = sum(counts.get(y + j, 0) for j in range(1, FWD_YEARS + 1))
        full = sum(counts.get(y + j, rate) if y + j <= last_full else rate
                   for j in range(1, FWD_YEARS + 1))
        out[y] = obs / full if full else 0.0
    return out


def panel_frame(cls: str, c8: pl.DataFrame, cov: dict[int, float]) -> pl.DataFrame:
    """Clean, in-box filings for one class with gate outcome and public flags."""
    sp = pl.read_parquet(PROC / f"surprise_class{cls}.parquet").filter(CLEAN).filter(
        pl.col("kl_vs_past").is_between(LO, HI)
        & pl.col("kl_vs_future").is_between(LO, HI)
    ).with_columns((pl.col("kl_vs_past") - pl.col("kl_vs_future")).alias("dkl"))
    tm = pl.read_parquet(PROC / f"tm_class{cls}.parquet",
                         columns=["serial_number", "registration_date"])
    d = sp.join(tm, on="serial_number", how="left").with_columns(
        pl.col("registration_date").str.strptime(pl.Date, "%Y%m%d", strict=False)
        .alias("reg_d"),
        pl.col("registration_date").str.slice(0, 4)
        .cast(pl.Int32, strict=False).alias("reg_year"),
    ).join(c8, on="serial_number", how="left").with_columns(
        ((pl.col("c8_d") - pl.col("reg_d")).dt.total_days() / 365.25).alias("c8_age")
    ).with_columns(
        (pl.col("reg_d").is_not_null()
         & pl.col("reg_year").is_between(REG_LO, REG_HI)).alias("gate_elapsed"),
        ((pl.col("c8_age") >= GATE_LO) & (pl.col("c8_age") < GATE_HI))
        .fill_null(False).alias("failed1"),
        pl.col("owner_name").map_elements(normalize, return_dtype=pl.Utf8).alias("norm"),
        pl.col("year").replace_strict(cov, default=0.0, return_dtype=pl.Float64)
        .alias("fwd_cov"),
    ).with_columns((pl.col("fwd_cov") >= COVERAGE_MIN).alias("full_window"))
    return d


def find_exemplar(d: pl.DataFrame, mark: str, year: int, owner: str) -> dict | None:
    """Match on mark text AND filing year AND owner substring -- all three."""
    cond = ((pl.col("year") == year)
            & pl.col("mark_identification").str.to_uppercase()
            .str.contains(f"^{mark}$|^{mark} "))
    if owner:
        cond = cond & pl.col("owner_name").str.to_uppercase().str.contains(owner)
    hits = d.filter(cond).sort("dkl", descending=True).head(1)
    return None if hits.is_empty() else hits.row(0, named=True)


# ---------------------------------------------------------------- drawing ---
def style_axes(ax, title: str, subtitle: str = "") -> None:
    ax.plot([LO, HI], [LO, HI], color=INK2, linewidth=0.7, linestyle="--", zorder=2)
    ax.text(HI - 0.12, HI - 0.12, "diagonal", fontsize=7.5, ha="right", va="top",
            color=INK2, zorder=3)
    ax.set_xlim(LO, HI)
    ax.set_ylim(LO, HI)
    ax.set_xlabel("KL against the PAST  (prospective; novelty at filing time)\n"
                  "higher $\\rightarrow$ less like the previous 5 years' filings",
                  fontsize=9, color=INK2)
    ax.set_ylabel("KL against the FUTURE  (retrospective; novelty in hindsight)\n"
                  "higher $\\rightarrow$ less like the next 5 years' filings",
                  fontsize=9, color=INK2)
    nlines = subtitle.count("\n") + 1 if subtitle else 0
    ax.set_title(title, fontsize=11, color=INK, pad=6 + 13 * nlines)
    if subtitle:
        ax.text(0.5, 1.012, subtitle, transform=ax.transAxes, ha="center",
                va="bottom", fontsize=8.3, color=INK2, linespacing=1.35)
    ax.grid(color=GRID, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8.5)
    ax.text(0.985, 0.03, "below diagonal:\nthe field moved\ntoward this filing",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
            color=INK2, zorder=3,
            bbox=dict(boxstyle="round,pad=0.28", fc="white", ec=GRID, alpha=0.85))
    ax.text(0.015, 0.97, "above diagonal:\nthe field had\nalready moved here",
            transform=ax.transAxes, va="top", fontsize=8, color=INK2, zorder=3,
            bbox=dict(boxstyle="round,pad=0.28", fc="white", ec=GRID, alpha=0.85))


def cell_counts(ax, x, y, gs) -> np.ndarray:
    """Point count per hex on the same grid, without drawing anything visible."""
    tmp = ax.hexbin(x, y, gridsize=gs, extent=(LO, HI, LO, HI), mincnt=1,
                    visible=False)
    c = np.asarray(tmp.get_array())
    tmp.remove()
    return c


def short_window_ring(ax, px, py) -> None:
    """Distinct, non-colour mark for an exemplar on a truncated forward window."""
    ax.plot([px], [py], linestyle="none", marker="o", mfc="none", mec=INK2,
            ms=15.0, mew=1.1, zorder=5)


def place_labels(ax, texts, xs, ys) -> None:
    if texts:
        adjust_text(texts, x=xs, y=ys, ax=ax, expand=(1.6, 1.9),
                    arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.5,
                                    shrinkA=2, shrinkB=2),
                    only_move={"text": "xy"},
                    force_text=(0.7, 0.9), force_static=(0.45, 0.45), max_move=22)


def smooth_density(x: np.ndarray, y: np.ndarray, bins: int = 90,
                   sigma: float = 1.8) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian-smoothed 2-D density on the plotting box, normalized to sum 1."""
    from scipy.ndimage import gaussian_filter
    h, xe, _ = np.histogram2d(x, y, bins=bins, range=[[LO, HI], [LO, HI]])
    h = gaussian_filter(h, sigma=sigma)
    return 0.5 * (xe[1:] + xe[:-1]), h / h.sum()


def hdr_levels(dens: np.ndarray, masses=(0.9, 0.75, 0.5)) -> list[float]:
    """Contour levels enclosing the given probability masses (highest-density regions)."""
    flat = np.sort(dens.ravel())[::-1]
    cum = np.cumsum(flat)
    return sorted(float(flat[np.searchsorted(cum, m)]) for m in masses)


def label_exemplar(ax, r, mark, year, xs, ys, texts) -> tuple[float, float, bool]:
    px, py = float(r["kl_vs_past"]), float(r["kl_vs_future"])
    short = not bool(r["full_window"])
    if short:
        short_window_ring(ax, px, py)
    xs.append(px)
    ys.append(py)
    texts.append(ax.text(px, py, f"{mark} ({year})" + (" $\\dag$" if short else ""),
                         fontsize=7.6, color=INK, zorder=7,
                         bbox=dict(boxstyle="round,pad=0.18", fc="white",
                                   ec="none", alpha=0.88)))
    return px, py, short


# ------------------------------------------------------------- variant A ----
def variant_a(frames: dict[str, pl.DataFrame], counts: list, exrows: list,
              diag: list) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15.6, 6.4))
    fig.patch.set_facecolor("white")
    hb = None
    for ax, (cls, label, examples) in zip(axes, PANELS):
        d = frames[cls]
        g = d.filter(pl.col("gate_elapsed"))
        x = g["kl_vs_past"].to_numpy()
        y = g["kl_vs_future"].to_numpy()
        f = g["failed1"].to_numpy().astype(float)
        base = float(f.mean())
        gs, mincnt = HEX_SURV[cls]
        n_short = int((~g["full_window"]).sum())
        counts.append({
            "variant": "A_survival", "cls": cls, "panel": label,
            "n_clean_in_box": d.height,
            "n_registered": int(d["reg_d"].is_not_null().sum()),
            "n_classifiable": g.height,
            "n_failed_gate": int(f.sum()), "n_survived_gate": int((1 - f).sum()),
            "pct_failed": round(100 * base, 2),
            "n_excluded_unclassifiable": d.height - g.height,
            "n_short_forward_window": n_short,
            "surface_year_cap": "none needed",
            "max_filing_year_on_surface": int(g["year"].max()),
        })
        log(f"[A/{cls}] classifiable {g.height:,} of {d.height:,} in-box "
            f"({100*g.height/d.height:.1f}%); failed {int(f.sum()):,} ({100*base:.1f}%); "
            f"short-forward-window filings on surface: {n_short} "
            f"(max filing year {int(g['year'].max())})")

        cnt = cell_counts(ax, x, y, gs)
        hb = ax.hexbin(x, y, C=f, gridsize=gs, extent=(LO, HI, LO, HI),
                       reduce_C_function=lambda v, b=base: 100.0 * (np.mean(v) - b),
                       mincnt=mincnt, cmap=DIVERGING,
                       norm=Normalize(vmin=-SURV_PP, vmax=SURV_PP),
                       linewidths=0.0, zorder=1)
        # One colourbar per panel. Each panel's colour scale is anchored to its
        # OWN base failure rate, so a single shared bar would make the reader
        # hold two different zero points at once; a per-panel bar also puts each
        # base rate where it is actually used.
        cbp = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.02)
        cbp.set_label(f"first-gate failure rate, percentage points\n"
                      f"from this panel's base of {100*base:.1f}%\n"
                      f"blue = more likely to survive",
                      fontsize=7.5, color=INK2)
        cbp.set_ticks([-16, -8, 0, 8, 16])
        cbp.set_ticklabels(["$-$16", "$-$8", "base", "+8", "+16"])
        cbp.ax.tick_params(labelsize=7, colors=INK2)
        cbp.outline.set_edgecolor(GRID)

        drawn = int((cnt >= mincnt).sum())
        vals = np.asarray(hb.get_array())
        clipped = int((np.abs(vals) > SURV_PP).sum())
        exp_sd = 100 * float(np.mean(np.sqrt(base * (1 - base) / cnt[cnt >= mincnt])))
        diag.append({
            "figure": "quadrant_survival", "cls": cls, "gridsize": gs,
            "mincnt": mincnt, "hex_cells_in_extent": int(len(cnt)),
            "cells_drawn": drawn, "cells_blank": int(len(cnt) - drawn),
            "pct_points_in_drawn_cells": round(
                100 * float(cnt[cnt >= mincnt].sum() / cnt.sum()), 1),
            "median_cell_n": int(np.median(cnt[cnt >= mincnt])),
            "base_rate": round(base, 4),
            "cell_sd": round(float(vals.std()), 3),
            "binomial_noise_sd": round(exp_sd, 3),
            "dispersion": round(float(vals.std()) / exp_sd, 2),
            "cells_clipped_at_colour_limit": clipped,
        })
        log(f"     hex gs={gs} mincnt={mincnt}: {drawn} cells drawn, "
            f"{len(cnt)-drawn} blank of {len(cnt)}; median cell n="
            f"{np.median(cnt[cnt>=mincnt]):.0f}; cell sd {vals.std():.2f}pp vs "
            f"{exp_sd:.2f}pp binomial noise (dispersion "
            f"{vals.std()/exp_sd:.2f}); {clipped} cells clipped")

        style_axes(ax, f"{label}  (class {cls})",
                   f"{g.height:,} classifiable registrations   |   "
                   f"{100*base:.1f}% failed the first gate   |   "
                   f"{drawn} hexes drawn")

        xs, ys, texts = [], [], []
        for mark, year, owner in examples:
            r = find_exemplar(d, mark, year, owner)
            if r is None:
                log(f"  miss [{cls}] {mark} ({year}) owner~{owner!r}")
                continue
            if r["gate_elapsed"]:
                state = "failed" if r["failed1"] else "survived"
            else:
                state = "not classifiable"
            spec = {"survived": dict(marker="o", mfc=BLUE, mec="white", ms=8.0),
                    "failed": dict(marker="s", mfc="none", mec=ORANGE, ms=8.0),
                    "not classifiable": dict(marker="^", mfc="white", mec=MUTED,
                                             ms=7.0)}[state]
            px, py, short = label_exemplar(ax, r, mark, year, xs, ys, texts)
            ax.plot([px], [py], linestyle="none", mew=1.5, zorder=Z_MARK, **spec)
            exrows.append({"variant": "A_survival", "cls": cls, "mark": mark,
                           "year": year, "owner_key": owner,
                           "owner_name": r["owner_name"],
                           "kl_vs_past": px, "kl_vs_future": py,
                           "dkl": px - py, "state": state,
                           "fwd_coverage": round(float(r["fwd_cov"]), 3),
                           "short_forward_window": short})
        place_labels(ax, texts, xs, ys)

    handles = [
        Line2D([], [], linestyle="none", marker="o", mfc=BLUE, mec="white",
               ms=8, mew=1.5, label="named mark: survived the gate"),
        Line2D([], [], linestyle="none", marker="s", mfc="none", mec=ORANGE,
               ms=8, mew=1.5, label="named mark: failed the gate"),
        Line2D([], [], linestyle="none", marker="^", mfc="white", mec=MUTED,
               ms=7, mew=1.5,
               label="named mark: not classifiable (unregistered or gate not yet elapsed)"),
        Line2D([], [], linestyle="none", marker="o", mfc="none", mec=INK2,
               ms=12, mew=1.1,
               label="$\\dag$ ringed: forward window shorter than 5 years, "
                     "vertical position not comparable"),
    ]
    fig.subplots_adjust(bottom=0.21, top=0.80, wspace=0.42)
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               fontsize=8.5, bbox_to_anchor=(0.5, 0.005))
    fig.suptitle("Vocabulary position and survival at the first Section 8 use-proof gate\n"
                 "registrations 2002-2017, the cohorts whose first gate has fully "
                 "elapsed; sparse cells left blank",
                 fontsize=12.5, color=INK, y=0.985)
    for ext in ("png", "pdf"):
        fig.savefig(RES / f"quadrant_survival.{ext}", dpi=170,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ------------------------------------------------------------- variant B ----
def variant_b(frames: dict[str, pl.DataFrame], counts: list, exrows: list) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15.6, 6.4))
    fig.patch.set_facecolor("white")
    rng = np.random.default_rng(7)
    for ax, (cls, label, examples) in zip(axes, PANELS):
        d = frames[cls]
        pub = d.filter(pl.col("is_pub"))
        counts.append({
            "variant": "B_public", "cls": cls, "panel": label,
            "n_clean_in_box": d.height,
            "n_public_wide": pub.height,
            "n_public_firms_wide": pub["norm"].n_unique(),
            "n_public_8a_only": int(d["is_8a"].sum()),
            "n_public_firms_8a_only": d.filter(pl.col("is_8a"))["norm"].n_unique(),
            "n_public_literal_formd": int(d["is_literal"].sum()),
            "pct_of_panel": round(100 * pub.height / d.height, 3),
        })
        log(f"[B/{cls}] public {pub.height:,} filings / {pub['norm'].n_unique():,} firms "
            f"({100*pub.height/d.height:.2f}% of panel); 8-A only "
            f"{int(d['is_8a'].sum()):,}; literal Form-D-gated {int(d['is_literal'].sum()):,}")

        # the restricted sample itself, as light texture under the contours
        px = pub["kl_vs_past"].to_numpy()
        py = pub["kl_vs_future"].to_numpy()
        shown = len(px)
        if shown > 8000:                       # subsample only where needed
            idx = rng.choice(shown, 8000, replace=False)
            px, py = px[idx], py[idx]
        ax.scatter(px, py, s=4, alpha=0.11, color=BLUE, linewidths=0, zorder=3,
                   rasterized=True)
        # public-subset density, dashed, in the second categorical slot
        gx2, hp = smooth_density(pub["kl_vs_past"].to_numpy(),
                                 pub["kl_vs_future"].to_numpy(), bins=60,
                                 sigma=2.4 if pub.height > 20000 else 3.4)
        ax.contour(gx2, gx2, hp.T, levels=hdr_levels(hp, (0.9, 0.75, 0.5)),
                   colors=[ORANGE], linewidths=1.3, linestyles="dashed", zorder=5)
        # context: smoothed density of the whole class, drawn OVER the scatter so
        # it stays visible where the restricted sample is dense
        gx, ctx = smooth_density(d["kl_vs_past"].to_numpy(),
                                 d["kl_vs_future"].to_numpy(), bins=60, sigma=2.4)
        ax.contour(gx, gx, ctx.T, levels=hdr_levels(ctx, (0.9, 0.75, 0.5)),
                   colors=[INK2], linewidths=0.9, alpha=0.85, zorder=5)

        mu_p = float(pub["dkl"].mean())
        mu_n = float(d.filter(~pl.col("is_pub"))["dkl"].mean())
        style_axes(ax, f"{label}  (class {cls})",
                   f"{pub.height:,} filings by {pub['norm'].n_unique():,} "
                   f"public-company owners   |   "
                   f"{100*pub.height/d.height:.1f}% of the {d.height:,} scored here\n"
                   f"mean $\\Delta$KL {mu_p:+.3f} against {mu_n:+.3f} for the rest "
                   f"of the class")

        xl, yl, texts = [], [], []
        for mark, year, owner in examples:
            r = find_exemplar(d, mark, year, owner)
            if r is None or not r["is_pub"]:
                exrows.append({"variant": "B_public", "cls": cls, "mark": mark,
                               "year": year, "owner_key": owner,
                               "owner_name": None if r is None else r["owner_name"],
                               "kl_vs_past": None if r is None else r["kl_vs_past"],
                               "kl_vs_future": None if r is None else r["kl_vs_future"],
                               "dkl": None if r is None else r["dkl"],
                               "state": "not in panel" if r is None
                               else "owner not public",
                               "fwd_coverage": None if r is None
                               else round(float(r["fwd_cov"]), 3),
                               "short_forward_window": None if r is None
                               else not bool(r["full_window"])})
                continue
            ex, ey, short = label_exemplar(ax, r, mark, year, xl, yl, texts)
            ax.plot([ex], [ey], linestyle="none", marker="o", mfc=ORANGE,
                    mec="white", ms=8.0, mew=1.4, zorder=Z_MARK)
            exrows.append({"variant": "B_public", "cls": cls, "mark": mark,
                           "year": year, "owner_key": owner,
                           "owner_name": r["owner_name"],
                           "kl_vs_past": ex, "kl_vs_future": ey,
                           "dkl": ex - ey, "state": "public",
                           "fwd_coverage": round(float(r["fwd_cov"]), 3),
                           "short_forward_window": short})
        place_labels(ax, texts, xl, yl)
        if shown > 8000:
            ax.text(0.015, 0.015, "8,000 of the points plotted, drawn at random",
                    transform=ax.transAxes, fontsize=7.8, color=INK2, va="bottom",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=GRID,
                              alpha=0.9))

    handles = [
        Line2D([], [], linestyle="none", marker="o", mfc=BLUE, mec="none", ms=6,
               alpha=0.6, label="filing by an owner that reached the public universe"),
        Line2D([], [], color=ORANGE, linestyle="--", linewidth=1.2,
               label="density of those filings"),
        Line2D([], [], color=INK2, linewidth=0.9, alpha=0.85,
               label="density of all scored filings in the class (context)"),
        Line2D([], [], linestyle="none", marker="o", mfc=ORANGE, mec="white", ms=8,
               mew=1.4, label="named mark whose owner qualifies"),
        Line2D([], [], linestyle="none", marker="o", mfc="none", mec=INK2, ms=12,
               mew=1.1, label="$\\dag$ ringed: forward window shorter than 5 years"),
    ]
    fig.subplots_adjust(bottom=0.21, top=0.80, wspace=0.42)
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               fontsize=8.5, bbox_to_anchor=(0.5, 0.005))
    fig.suptitle("Vocabulary position of filings whose owner reached the public-company universe\n"
                 "public = CIK in the SEC Financial Statement Data Sets or carrying an "
                 "EDGAR 8-A registration; owners matched on normalized name",
                 fontsize=12.5, color=INK, y=0.985)
    for ext in ("png", "pdf"):
        fig.savefig(RES / f"quadrant_public.{ext}", dpi=170,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ------------------------------------------------------------- variant C ----
def variant_c(frames: dict[str, pl.DataFrame], counts: list, exrows: list,
              diag: list) -> None:
    """SEC-listing rate surface, log2 of the ratio to each panel's base rate."""
    fig, axes = plt.subplots(1, 2, figsize=(15.6, 6.4))
    fig.patch.set_facecolor("white")
    hb = None
    for ax, (cls, label, examples) in zip(axes, PANELS):
        d = frames[cls]
        # Conditioned on grant, as the paper's headline results are: an
        # ungranted application never reached the market, so including it
        # mixes examination selection into a market outcome.
        s = d.filter(pl.col("year").is_between(SURF_LO, SURF_HI)
                     & pl.col("reg_d").is_not_null())
        x = s["kl_vs_past"].to_numpy()
        y = s["kl_vs_future"].to_numpy()
        f = s["is_pub"].to_numpy().astype(float)
        base = float(f.mean())
        gs, mincnt = HEX_LIST[cls]
        counts.append({
            "variant": "C_listing", "cls": cls, "panel": label,
            "n_clean_in_box": d.height,
            "n_on_surface": s.height,
            "surface_year_cap": f"{SURF_LO}-{SURF_HI}",
            "surface_conditioned_on": "granted (registration_date present)",
            "n_excluded_short_window": int((d["year"] > SURF_HI).sum()),
            "n_excluded_preburnin": int((d["year"] < SURF_LO).sum()),
            "n_listed_wide": int(f.sum()),
            "n_listed_firms_wide": s.filter(pl.col("is_pub"))["norm"].n_unique(),
            "base_rate_pct": round(100 * base, 4),
            "n_listed_8a_only": int(s["is_8a"].sum()),
            "base_rate_8a_only_pct": round(100 * float(s["is_8a"].mean()), 4),
            "n_listed_literal_formd": int(s["is_literal"].sum()),
        })
        log(f"[C/{cls}] surface {s.height:,} filings {SURF_LO}-{SURF_HI} "
            f"(dropped {int((d['year']>SURF_HI).sum()):,} short-window, "
            f"{int((d['year']<SURF_LO).sum()):,} pre-burn-in); listed "
            f"{int(f.sum()):,} = {100*base:.3f}% by "
            f"{s.filter(pl.col('is_pub'))['norm'].n_unique():,} firms; "
            f"8-A only {int(s['is_8a'].sum()):,} = {100*float(s['is_8a'].mean()):.3f}%")

        cnt = cell_counts(ax, x, y, gs)

        def lr(v, b=base):
            m = float(np.mean(v))
            return np.log2(max(m, b / 64.0) / b)

        hb = ax.hexbin(x, y, C=f, gridsize=gs, extent=(LO, HI, LO, HI),
                       reduce_C_function=lr, mincnt=mincnt, cmap=DIVERGING_R,
                       norm=Normalize(vmin=-LIST_L2, vmax=LIST_L2),
                       linewidths=0.0, zorder=1)
        # One colourbar per panel, as in variant A: the scale is a multiple of
        # THIS panel's base rate, and the two panels' bases differ by more than
        # a factor of two, so a shared bar would be genuinely ambiguous.
        cbp = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.02)
        cbp.set_label(f"listing rate as a multiple of this\n"
                      f"panel's base of {100*base:.2f}% (log scale)\n"
                      f"blue = more likely to be listed",
                      fontsize=7.5, color=INK2)
        cbp.set_ticks([-2.0, -1.0, 0.0, 1.0, 2.0])
        cbp.set_ticklabels(["0.25$\\times$", "0.5$\\times$", "base",
                            "2$\\times$", "4$\\times$"])
        cbp.ax.tick_params(labelsize=7, colors=INK2)
        cbp.outline.set_edgecolor(GRID)

        drawn = int((cnt >= mincnt).sum())
        vals = np.asarray(hb.get_array())
        clipped = int((np.abs(vals) > LIST_L2).sum())
        # spread expected from binomial noise alone, in the same log2 units
        exp_sd = float(np.sqrt(np.mean(
            (base * (1 - base) / cnt[cnt >= mincnt]) / base ** 2))) / np.log(2)
        disp = float(vals.std()) / exp_sd
        diag.append({
            "figure": "quadrant_listing", "cls": cls, "gridsize": gs,
            "mincnt": mincnt, "hex_cells_in_extent": int(len(cnt)),
            "cells_drawn": drawn, "cells_blank": int(len(cnt) - drawn),
            "pct_points_in_drawn_cells": round(
                100 * float(cnt[cnt >= mincnt].sum() / cnt.sum()), 1),
            "median_cell_n": int(np.median(cnt[cnt >= mincnt])),
            "base_rate": round(base, 5),
            "cell_sd": round(float(vals.std()), 3),
            "binomial_noise_sd": round(exp_sd, 3),
            "dispersion": round(disp, 2),
            "cells_clipped_at_colour_limit": clipped,
        })
        log(f"     hex gs={gs} mincnt={mincnt}: {drawn} cells drawn, "
            f"{len(cnt)-drawn} blank of {len(cnt)}; median cell n="
            f"{np.median(cnt[cnt>=mincnt]):.0f}; log2 spread sd {vals.std():.3f} vs "
            f"{exp_sd:.3f} from binomial noise (dispersion {disp:.2f}); "
            f"{clipped} cells clipped at +/-{LIST_L2}")

        # marginal gradients, printed so the "is it flat?" question is answerable
        # from numbers as well as from the picture
        for col in ("dkl", "kl_vs_past", "kl_vs_future"):
            a = s[col].to_numpy()
            cuts = np.quantile(a, [.2, .4, .6, .8])
            qi = np.searchsorted(cuts, a)
            rates = [100 * f[qi == q].mean() for q in range(5)]
            log("       " + f"{col:12s} listing rate by quintile: "
                + " ".join(f"{v:.3f}%" for v in rates)
                + f"   Q5/Q1 = {rates[4]/rates[0]:.2f}")
            counts[-1][f"q5q1_{col}"] = round(rates[4] / rates[0], 3)

        # Is the gradient just filing-era composition?  Recompute the dKL
        # quintile contrast WITHIN filing year, then average over years weighted
        # by year size.  If the within-year contrast matches the pooled one the
        # surface is not a vintage artefact.
        yrs = s["year"].to_numpy()
        a = s["dkl"].to_numpy()
        num_q1 = num_q5 = den_q1 = den_q5 = 0.0
        for yy in np.unique(yrs):
            m = yrs == yy
            if m.sum() < 500:
                continue
            av = a[m]
            cuts = np.quantile(av, [.2, .8])
            fq1 = f[m][av <= cuts[0]]
            fq5 = f[m][av >= cuts[1]]
            num_q1 += fq1.sum(); den_q1 += len(fq1)
            num_q5 += fq5.sum(); den_q5 += len(fq5)
        wq1, wq5 = num_q1 / den_q1, num_q5 / den_q5
        log(f"       within-filing-year dKL contrast: Q1 {100*wq1:.3f}%  "
            f"Q5 {100*wq5:.3f}%   Q5/Q1 = {wq5/wq1:.2f}  "
            f"(pooled {counts[-1]['q5q1_dkl']:.2f})")
        counts[-1]["q5q1_dkl_within_year"] = round(wq5 / wq1, 3)

        style_axes(ax, f"{label}  (class {cls})",
                   f"{s.height:,} filings {SURF_LO}-{SURF_HI}   |   base rate "
                   f"{100*base:.2f}% ({int(f.sum()):,} filings, "
                   f"{s.filter(pl.col('is_pub'))['norm'].n_unique():,} owners)\n"
                   f"cell-to-cell spread {disp:.1f}$\\times$ what sampling noise "
                   f"alone would produce   |   {drawn} hexes drawn")

        xs, ys, texts = [], [], []
        for mark, year, owner in examples:
            r = find_exemplar(d, mark, year, owner)
            if r is None:
                log(f"  miss [{cls}] {mark} ({year}) owner~{owner!r}")
                continue
            listed = bool(r["is_pub"])
            spec = (dict(marker="o", mfc=INK, mec="white", ms=7.5) if listed
                    else dict(marker="D", mfc="none", mec=INK, ms=6.5))
            px, py, short = label_exemplar(ax, r, mark, year, xs, ys, texts)
            ax.plot([px], [py], linestyle="none", mew=1.4, zorder=Z_MARK, **spec)
            exrows.append({"variant": "C_listing", "cls": cls, "mark": mark,
                           "year": year, "owner_key": owner,
                           "owner_name": r["owner_name"],
                           "kl_vs_past": px, "kl_vs_future": py, "dkl": px - py,
                           "state": "owner listed" if listed else "owner not listed",
                           "fwd_coverage": round(float(r["fwd_cov"]), 3),
                           "short_forward_window": short})
        place_labels(ax, texts, xs, ys)

    handles = [
        Line2D([], [], linestyle="none", marker="o", mfc=INK, mec="white", ms=7.5,
               mew=1.4, label="named mark: owner in the listed universe"),
        Line2D([], [], linestyle="none", marker="D", mfc="none", mec=INK, ms=6.5,
               mew=1.4, label="named mark: owner not in the listed universe"),
        Line2D([], [], linestyle="none", marker="o", mfc="none", mec=INK2, ms=12,
               mew=1.1,
               label="$\\dag$ ringed: filed after 2020, forward window shorter than "
                     "5 years and excluded from the surface"),
    ]
    fig.subplots_adjust(bottom=0.21, top=0.80, wspace=0.42)
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               fontsize=8.5, bbox_to_anchor=(0.5, 0.005))
    fig.suptitle("Vocabulary position and the chance the owner reached the listed universe\n"
                 "granted marks filed 1995-2020, the years with a complete five-year forward window;\n"
                 "listed = owner CIK in the SEC Financial Statement Data Sets or "
                 "carrying an EDGAR 8-A registration",
                 fontsize=12.5, color=INK, y=0.995)
    for ext in ("png", "pdf"):
        fig.savefig(RES / f"quadrant_listing.{ext}", dpi=170,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> int:
    log("[load] first C8 event per serial")
    c8 = c8_first()
    wide, only8a, literal = public_keys()
    frames = {}
    for cls, label, _ in PANELS:
        cov = forward_coverage(cls)
        log(f"[coverage {cls}] " + "  ".join(
            f"{y}:{cov[y]:.2f}" for y in sorted(cov) if y >= 2018))
        frames[cls] = panel_frame(cls, c8, cov).with_columns(
            pl.col("norm").is_in(list(wide)).fill_null(False).alias("is_pub"),
            pl.col("norm").is_in(list(only8a)).fill_null(False).alias("is_8a"),
            pl.col("owner_name").is_in(list(literal)).fill_null(False).alias("is_literal"),
        )
        log(f"[panel {cls}] {frames[cls].height:,} clean filings inside the axis box; "
            f"{int((~frames[cls]['full_window']).sum()):,} on a short forward window "
            f"({100*float((~frames[cls]['full_window']).mean()):.1f}%)")
    counts: list[dict] = []
    exrows: list[dict] = []
    diag: list[dict] = []
    variant_a(frames, counts, exrows, diag)
    variant_b(frames, counts, exrows)
    variant_c(frames, counts, exrows, diag)
    pl.from_dicts(counts, infer_schema_length=None).write_csv(
        RES / "quadrant_outcome_counts.csv")
    pl.from_dicts(exrows, infer_schema_length=None).write_csv(
        RES / "quadrant_outcome_exemplars.csv")
    pl.from_dicts(diag).write_csv(RES / "quadrant_surface_diagnostics.csv")
    log("[done] quadrant_survival / quadrant_public / quadrant_listing .{png,pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
