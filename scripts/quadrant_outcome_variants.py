"""Two outcome-encoded variants of the quadrant scatter (Figure 1).

The published quadrant figure (scripts/quadrant_regen.py -> paper/results/quadrant.png)
shows WHERE a filing sits in vocabulary space but says nothing about what happened
to it.  This script produces two variants on the same axes and the same two class
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
    "Public" is CIK-keyed, never owner-name-keyed: a CIK counts if it appears in
    data/processed/sec_firm_year.parquet (the SEC Financial Statement Data Sets =
    the XBRL filer universe) or carries an EDGAR 8-A12B/8-A12G registration
    (`in_8a` in funding_owner_match.parquet = the actual moment of exchange
    listing).  Owner strings are then attached to those CIKs through BOTH
    name->CIK tables (funding_owner_match.parquet and uspto_sec_crosswalk.parquet)
    and matched on the shared normalized key `norm`, not on the raw string, so a
    firm flagged under one USPTO spelling is flagged under all of them.
    This deliberately does NOT use `in_sec` (name-keyed, internally inconsistent
    across 1,912 CIKs) and does NOT use `in_fsds`/`in_8a` membership alone:
    every row of funding_owner_match is a Reg D (Form D) match, so those markers
    silently restrict the "public" universe to firms that also filed a Form D
    notice in 2009q1-2026q2.  Under that literal restriction class 032 loses
    Coca-Cola and PepsiCo and class 009 loses Apple and IBM.  Both definitions
    are counted and reported; the wider one is the one plotted.

Overplotting.  Both panels carry 10^5-10^6 eligible filings and the two outcome
groups occupy almost exactly the same region, so a two-colour scatter of raw
points is mush at any alpha.  Both variants therefore draw a hexbin RATE SURFACE
(mean outcome per hex, cells below a minimum count left blank) plus a small
foreground layer of named exemplars.  Variant B additionally scatters the
restricted sample itself, since it is small enough to show.

Outputs: paper/results/quadrant_survival.{png,pdf}
         paper/results/quadrant_public.{png,pdf}
         paper/results/quadrant_outcome_counts.csv
         paper/results/quadrant_outcome_exemplars.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
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

PANELS = [
    ("009", "Software & Electronics", [
        ("OPENAI", 2016), ("CLAUDE", 2023), ("INSTAGRAM", 2011),
        ("ETHEREUM", 2018), ("KUBERNETES", 2014), ("CHATGPT", 2022),
        # WINDOWS (1995) dropped: mark text and year resolve to a third-party
        # registration owned by Softblox Incorporated, not to Microsoft.
        # Exemplars match on text and year only, so check owner_name before
        # adding a common-word mark here.
        ("UBER", 2014), ("PHOTOSHOP", 2003),
        ("AIRPODS", 2015), ("KINDLE", 2010), ("IPOD", 2001),
    ]),
    ("032", "Beer & Soft Drinks", [
        ("LIQUID DEATH", 2023), ("WHITE CLAW SELTZER WORKS", 2016),
        ("ROCKSTAR", 2002), ("HARD MTN DEW", 2021), ("FIJI", 2005),
        ("SMARTWATER", 1997), ("KOMBUCHA", 1997), ("POWERADE", 1996),
        ("GATORADE", 1996), ("RED BULL", 1995), ("MONSTER ENERGY", 2002),
        ("CELSIUS", 2004),
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
HEX = {"009": (26, 250), "032": (14, 120)}  # (gridsize, min points per hex)


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


def panel_frame(cls: str, c8: pl.DataFrame) -> pl.DataFrame:
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
    )
    return d


def find_exemplar(d: pl.DataFrame, mark: str, year: int) -> dict | None:
    hits = d.filter(
        (pl.col("year") == year)
        & pl.col("mark_identification").str.to_uppercase()
        .str.contains(f"^{mark}$|^{mark} ")
    ).sort("dkl", descending=True).head(1)
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


def place_labels(ax, texts, xs, ys) -> None:
    if texts:
        adjust_text(texts, x=xs, y=ys, ax=ax, expand=(1.6, 1.9),
                    arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.5,
                                    shrinkA=2, shrinkB=2),
                    only_move={"text": "xy"},
                    force_text=(0.7, 0.9), force_static=(0.45, 0.45), max_move=22)


# ------------------------------------------------------------- variant A ----
def variant_a(frames: dict[str, pl.DataFrame], counts: list, exrows: list) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 6.4))
    fig.patch.set_facecolor("white")
    hb = None
    for ax, (cls, label, examples) in zip(axes, PANELS):
        d = frames[cls]
        g = d.filter(pl.col("gate_elapsed"))
        x = g["kl_vs_past"].to_numpy()
        y = g["kl_vs_future"].to_numpy()
        f = g["failed1"].to_numpy().astype(float)
        base = float(f.mean())
        gs, mincnt = HEX[cls]
        counts.append({
            "variant": "A_survival", "cls": cls, "panel": label,
            "n_clean_in_box": d.height,
            "n_registered": int(d["reg_d"].is_not_null().sum()),
            "n_classifiable": g.height,
            "n_failed_gate": int(f.sum()), "n_survived_gate": int((1 - f).sum()),
            "pct_failed": round(100 * base, 2),
            "n_excluded_unclassifiable": d.height - g.height,
        })
        log(f"[A/{cls}] classifiable {g.height:,} of {d.height:,} in-box "
            f"({100*g.height/d.height:.1f}%); failed {int(f.sum()):,} ({100*base:.1f}%)")

        hb = ax.hexbin(x, y, C=f, reduce_C_function=np.mean, gridsize=gs,
                       extent=(LO, HI, LO, HI), mincnt=mincnt, cmap=DIVERGING,
                       norm=TwoSlopeNorm(vmin=base - 0.16, vcenter=base,
                                         vmax=base + 0.16),
                       linewidths=0.0, zorder=1)
        style_axes(ax, f"{label}  (class {cls})",
                   f"{g.height:,} classifiable registrations   |   "
                   f"{100*base:.1f}% failed the first gate")

        xs, ys, texts = [], [], []
        for mark, year in examples:
            r = find_exemplar(d, mark, year)
            if r is None:
                log(f"  miss [{cls}] {mark} ({year})")
                continue
            px, py = float(r["kl_vs_past"]), float(r["kl_vs_future"])
            if r["gate_elapsed"]:
                state = "failed" if r["failed1"] else "survived"
            else:
                state = "not classifiable"
            spec = {"survived": dict(marker="o", mfc=BLUE, mec="white", ms=8.0),
                    "failed": dict(marker="s", mfc="none", mec=ORANGE, ms=8.0),
                    "not classifiable": dict(marker="^", mfc="white", mec=MUTED,
                                             ms=7.0)}[state]
            ax.plot([px], [py], linestyle="none", mew=1.5, zorder=6, **spec)
            xs.append(px)
            ys.append(py)
            texts.append(ax.text(px, py, f"{mark} ({year})", fontsize=7.6,
                                 color=INK, zorder=7,
                                 bbox=dict(boxstyle="round,pad=0.18", fc="white",
                                           ec="none", alpha=0.88)))
            exrows.append({"variant": "A_survival", "cls": cls, "mark": mark,
                           "year": year, "kl_vs_past": px, "kl_vs_future": py,
                           "dkl": px - py, "state": state})
        place_labels(ax, texts, xs, ys)

    handles = [
        Line2D([], [], linestyle="none", marker="o", mfc=BLUE, mec="white",
               ms=8, mew=1.5, label="named mark: survived the gate"),
        Line2D([], [], linestyle="none", marker="s", mfc="none", mec=ORANGE,
               ms=8, mew=1.5, label="named mark: failed the gate"),
        Line2D([], [], linestyle="none", marker="^", mfc="white", mec=MUTED,
               ms=7, mew=1.5,
               label="named mark: not classifiable (unregistered or gate not yet elapsed)"),
    ]
    fig.subplots_adjust(bottom=0.19, top=0.80, right=0.90)
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=8.5, bbox_to_anchor=(0.5, 0.03))
    cax = fig.add_axes([0.925, 0.30, 0.014, 0.42])
    cb = fig.colorbar(hb, cax=cax)
    cb.set_label("share of registrations in the cell that FAILED the first\n"
                 "Section 8 gate, centred on that panel's own base rate",
                 fontsize=8, color=INK2)
    cb.ax.tick_params(labelsize=7.5, colors=INK2)
    cb.outline.set_edgecolor(GRID)
    fig.suptitle("Vocabulary position and survival at the first Section 8 use-proof gate\n"
                 "registrations 2002-2017, the cohorts whose first gate has fully "
                 "elapsed; sparse cells left blank",
                 fontsize=12.5, color=INK, y=0.985)
    for ext in ("png", "pdf"):
        fig.savefig(RES / f"quadrant_survival.{ext}", dpi=170,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ------------------------------------------------------------- variant B ----
def variant_b(frames: dict[str, pl.DataFrame], wide: set[str], only8a: set[str],
              literal: set[str], counts: list, exrows: list) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 6.4))
    fig.patch.set_facecolor("white")
    rng = np.random.default_rng(7)
    for ax, (cls, label, examples) in zip(axes, PANELS):
        d = frames[cls].with_columns(
            pl.col("norm").is_in(list(wide)).alias("is_pub"),
            pl.col("norm").is_in(list(only8a)).alias("is_8a"),
            pl.col("owner_name").is_in(list(literal)).alias("is_literal"),
        )
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
        for mark, year in examples:
            r = find_exemplar(d, mark, year)
            if r is None or not r["is_pub"]:
                exrows.append({"variant": "B_public", "cls": cls, "mark": mark,
                               "year": year,
                               "kl_vs_past": None if r is None else r["kl_vs_past"],
                               "kl_vs_future": None if r is None else r["kl_vs_future"],
                               "dkl": None if r is None else r["dkl"],
                               "state": "not in panel" if r is None
                               else f"owner not public ({r['owner_name']})"})
                continue
            ex, ey = float(r["kl_vs_past"]), float(r["kl_vs_future"])
            ax.plot([ex], [ey], linestyle="none", marker="o", mfc=ORANGE,
                    mec="white", ms=8.0, mew=1.4, zorder=6)
            xl.append(ex)
            yl.append(ey)
            texts.append(ax.text(ex, ey, f"{mark} ({year})", fontsize=7.6,
                                 color=INK, zorder=7,
                                 bbox=dict(boxstyle="round,pad=0.18", fc="white",
                                           ec="none", alpha=0.88)))
            exrows.append({"variant": "B_public", "cls": cls, "mark": mark,
                           "year": year, "kl_vs_past": ex, "kl_vs_future": ey,
                           "dkl": ex - ey, "state": f"public: {r['owner_name']}"})
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
    ]
    fig.subplots_adjust(bottom=0.19, top=0.80)
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               fontsize=8.5, bbox_to_anchor=(0.5, 0.015))
    fig.suptitle("Vocabulary position of filings whose owner reached the public-company universe\n"
                 "public = CIK in the SEC Financial Statement Data Sets or carrying an "
                 "EDGAR 8-A registration; owners matched on normalized name",
                 fontsize=12.5, color=INK, y=0.985)
    for ext in ("png", "pdf"):
        fig.savefig(RES / f"quadrant_public.{ext}", dpi=170,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> int:
    log("[load] first C8 event per serial")
    c8 = c8_first()
    wide, only8a, literal = public_keys()
    frames = {}
    for cls, label, _ in PANELS:
        frames[cls] = panel_frame(cls, c8)
        log(f"[panel {cls}] {frames[cls].height:,} clean filings inside the axis box")
    counts: list[dict] = []
    exrows: list[dict] = []
    variant_a(frames, counts, exrows)
    variant_b(frames, wide, only8a, literal, counts, exrows)
    pl.from_dicts(counts).write_csv(RES / "quadrant_outcome_counts.csv")
    pl.from_dicts(exrows).write_csv(RES / "quadrant_outcome_exemplars.csv")
    log("[done] quadrant_survival.{png,pdf}, quadrant_public.{png,pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
