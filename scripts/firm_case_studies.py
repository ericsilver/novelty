"""Firm case studies: patent->product->trademark pipeline via DeltaKL.

For each of 8 IP-heavy firms, we build:
  1. The yearly patent grant count from PatentsView
     (g_patent.tsv, g_assignee_disambiguated.tsv), with the top CPC subclass
     mix as a content tag for the portfolio.
  2. The yearly trademark filing count from USPTO tm_class*.parquet,
     joined with surprise_class*.parquet to attach prospective_kl,
     retrospective_kl, and DeltaKL = prospective_kl - retrospective_kl per
     filing.
  3. A small hand-curated set of *anchor marks* per firm
     (POWERWALL, MODEL S, AIRBLADE, KINDLE, ALEXA, CUDA, RTX, ...)
     whose DeltaKL we report directly --- these are the closest things
     to "trademark filings that look like they correspond to a specific
     patented product."
  4. The firm's mean DeltaKL trajectory across years, alongside the
     patent grant timeline, so we can eyeball whether vocabulary-novelty
     years coincide with major product launches that themselves should
     trail patent activity by some lag.

CAVEATS encoded throughout:
  * PatentsView "patent_date" is the GRANT date. The patent that
    *covers* a product is usually filed several years before the
    product launches, so the trademark filing can easily *precede*
    the grant date of the relevant patent.
  * We do NOT attempt patent->trademark matching at the document
    level. The patent timeline is a *portfolio-activity* signal, not
    a "this patent maps to this mark" claim. Where we name a specific
    patent in the LaTeX writeup, it is from public reporting, not from
    our data.
  * Tesla's PatentsView assignee bucket is contaminated by unrelated
    "Tesla *" firms; the regex is tightened to Tesla Motors / Tesla, Inc.
    Apple, Amazon, NVIDIA, Salesforce, Dyson, Pfizer, Moderna are
    cleaner. See FIRMS regex blocks.

Outputs:
  paper/results/firm_case_studies_summary.csv
  paper/results/case_study_{firm}.txt           (one per firm)
  paper/results/case_study_{firm}_timeline.csv  (one per firm)
  paper/results/firm_case_studies.png           (multi-panel figure)
"""

from __future__ import annotations

import gc
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parents[1]
PV_DIR = REPO / "data" / "raw" / "patentsview"
PROC = REPO / "data" / "processed"
RESULTS = REPO / "paper" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Firms
# ---------------------------------------------------------------------
@dataclass
class Firm:
    key: str               # short slug, used in filenames
    label: str             # display name
    tm_regex: str          # regex on trademark owner_name
    pv_regex: str          # regex on PatentsView disambig_assignee_organization
    # Hand-curated anchor marks: (brand, ex-ante product story, year-of-product-launch)
    # These are well-documented public facts; we use them only to *look up*
    # the filing in our data, not to invent linkages.
    anchors: list[tuple[str, str, int]]


FIRMS: list[Firm] = [
    Firm(
        key="apple",
        label="Apple",
        tm_regex=r"(?i)^apple\s+(inc|computer)\.?,?$|^apple\s+inc\.?$",
        pv_regex=r"(?i)^apple\s+(inc|computer)",
        anchors=[
            ("IPOD",        "iPod launched Oct 2001",                       2001),
            ("IPHONE",      "iPhone launched Jun 2007",                     2007),
            ("IPAD",        "iPad launched Apr 2010",                       2010),
            ("SIRI",        "Siri shipped with iPhone 4S Oct 2011",         2011),
            ("APPLE WATCH", "Apple Watch launched Apr 2015",                2015),
            ("AIRPODS",     "AirPods launched Dec 2016",                    2016),
            ("HOMEPOD",     "HomePod launched Feb 2018",                    2018),
            ("VISION PRO",  "Vision Pro launched Feb 2024",                 2024),
        ],
    ),
    Firm(
        key="tesla",
        label="Tesla",
        tm_regex=r"(?i)^tesla,?\s*inc\.?$|^tesla\s+motors,?\s*inc\.?$",
        pv_regex=r"(?i)^(tesla\s+motors,?\s*inc|tesla,\s*inc)\.?$",
        anchors=[
            ("MODEL S",     "Model S launched Jun 2012",                    2012),
            ("MODEL X",     "Model X launched Sep 2015",                    2015),
            ("MODEL 3",     "Model 3 announced Mar 2016, ship Jul 2017",    2017),
            ("POWERWALL",   "Powerwall announced Apr 2015",                 2015),
            ("POWERPACK",   "Powerpack commercial battery 2015",            2015),
            ("MEGAPACK",    "Megapack announced Jul 2019",                  2019),
            ("CYBERTRUCK",  "Cybertruck unveiled Nov 2019, ship Nov 2023",  2023),
            ("AUTOPILOT",   "Autopilot v1 Oct 2015",                        2015),
            ("ROBOTAXI",    "Robotaxi/Cybercab unveil Oct 2024",            2024),
            ("CYBERCAB",    "Robotaxi/Cybercab unveil Oct 2024",            2024),
        ],
    ),
    Firm(
        key="nvidia",
        label="NVIDIA",
        tm_regex=r"(?i)^nvidia\s+corporation\.?$",
        pv_regex=r"(?i)^nvidia\s+corporation\.?$",
        anchors=[
            ("GEFORCE",     "GeForce 256 launched Oct 1999",                1999),
            ("CUDA",        "CUDA released Jun 2007",                       2007),
            ("TEGRA",       "Tegra SoC announced 2008",                     2008),
            ("G-SYNC",      "G-Sync introduced Oct 2013",                   2013),
            ("DGX",         "DGX-1 announced Apr 2016",                     2016),
            ("RTX",         "GeForce RTX 20 series Aug 2018",               2018),
            ("JETSON",      "Jetson AGX Xavier announced Jun 2018",         2018),
            ("OMNIVERSE",   "Omniverse public beta Dec 2020",               2020),
        ],
    ),
    Firm(
        key="pfizer",
        label="Pfizer",
        tm_regex=r"(?i)^pfizer\s+(inc\.?|products\s+inc\.?)$|^pfizer\s+inc$",
        pv_regex=r"(?i)^pfizer\s+(inc|products)",
        anchors=[
            ("VIAGRA",      "Viagra approved Mar 1998",                     1998),
            ("CELEBREX",    "Celebrex approved Dec 1998 (co-marketed)",     1998),
            ("LIPITOR",     "Lipitor approved 1996, Pfizer marketed",       1996),
            ("LYRICA",      "Lyrica approved Dec 2004",                     2004),
            ("CHANTIX",     "Chantix approved May 2006",                    2006),
            ("XELJANZ",     "Xeljanz approved Nov 2012",                    2012),
            ("PREVNAR",     "Prevnar/Prevnar 13/20",                        2010),
            ("COMIRNATY",   "Comirnaty (BNT162b2) EUA Dec 2020",            2020),
            ("PAXLOVID",    "Paxlovid EUA Dec 2021",                        2021),
        ],
    ),
    Firm(
        key="moderna",
        label="Moderna",
        tm_regex=r"(?i)^(modernatx|moderna\s+therapeutics),?\s*inc\.?$",
        pv_regex=r"(?i)^(modernatx|moderna\s+therapeutics),?\s*inc\.?$",
        anchors=[
            ("MODERNA",     "Moderna Therapeutics name 2010-",              2010),
            ("SPIKEVAX",    "Spikevax (mRNA-1273) brand 2021",              2021),
            ("MRESVIA",     "mResvia RSV vaccine approved May 2024",        2024),
        ],
    ),
    Firm(
        key="dyson",
        label="Dyson",
        tm_regex=r"(?i)^dyson\s+(technology|limited)\b",
        pv_regex=r"(?i)^dyson\s+(technology|limited)\b",
        anchors=[
            ("DUAL CYCLONE","Dual Cyclone bagless vacuum 1993-",            1993),
            ("AIRBLADE",    "Airblade hand dryer launched 2006",            2006),
            ("AIRWRAP",     "Airwrap launched 2018",                        2018),
            ("SUPERSONIC",  "Supersonic hair dryer launched 2016",          2016),
        ],
    ),
    Firm(
        key="amazon",
        label="Amazon",
        tm_regex=r"(?i)^amazon\s+technologies,?\s*inc\.?$",
        pv_regex=r"(?i)^amazon\s+technologies",
        anchors=[
            ("KINDLE",      "Kindle launched Nov 2007",                     2007),
            ("FIRE TV",     "Fire TV launched Apr 2014",                    2014),
            ("ECHO",        "Echo launched Nov 2014",                       2014),
            ("ALEXA",       "Alexa shipped with Echo Nov 2014",             2014),
            ("PRIME AIR",   "Prime Air drone announced Dec 2013",           2013),
            ("BLINK",       "Blink cameras (acquired 2017)",                2017),
            ("RING",        "Ring (acquired 2018)",                         2018),
        ],
    ),
    Firm(
        key="salesforce",
        label="Salesforce",
        tm_regex=r"(?i)^salesforce(\.com)?,?\s*(inc|llc)\.?$",
        pv_regex=r"(?i)^salesforce(\.com)?,?\s*inc\.?$",
        anchors=[
            ("CHATTER",     "Chatter social-collab launched 2010",          2010),
            ("HEROKU",      "Heroku (acquired Dec 2010)",                   2010),
            ("PARDOT",      "Pardot (acquired Jun 2013)",                   2013),
            ("MULESOFT",    "MuleSoft (acquired May 2018)",                 2018),
            ("TABLEAU",     "Tableau (acquired Aug 2019)",                  2019),
            ("TRAILHEAD",   "Trailhead launched 2014",                      2014),
            ("SLACK",       "Slack (acquired Jul 2021)",                    2021),
        ],
    ),
]


# ---------------------------------------------------------------------
# 1. Patent timeline (firm-year)
# ---------------------------------------------------------------------
def load_patent_data(verbose: bool = True) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return (assignee_df, patent_year_df) joined to firm regexes."""
    if verbose:
        print("[load] g_assignee_disambiguated.tsv …", file=sys.stderr, flush=True)
    ag = pl.scan_csv(
        PV_DIR / "g_assignee_disambiguated.tsv", separator="\t",
        schema_overrides={"patent_id": pl.Utf8, "assignee_id": pl.Utf8},
        ignore_errors=True,
    ).select("patent_id", "disambig_assignee_organization").filter(
        pl.col("disambig_assignee_organization").is_not_null()
    ).collect()
    if verbose:
        print(f"       {ag.height:,} assignee rows", file=sys.stderr)

    if verbose:
        print("[load] g_patent.tsv …", file=sys.stderr, flush=True)
    pat = pl.scan_csv(
        PV_DIR / "g_patent.tsv", separator="\t",
        schema_overrides={"patent_id": pl.Utf8, "patent_date": pl.Utf8},
        ignore_errors=True,
    ).select("patent_id", "patent_date").collect()
    pat = pat.with_columns(
        pl.col("patent_date").str.slice(0, 4).cast(pl.Int32, strict=False).alias("year")
    ).drop_nulls("year")
    if verbose:
        print(f"       {pat.height:,} patents", file=sys.stderr)

    return ag, pat


def firm_patent_timeline(
    ag: pl.DataFrame, pat: pl.DataFrame, firm: Firm
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return (year-count DF, patent_id list with year & cpc) for a firm."""
    m = ag.filter(pl.col("disambig_assignee_organization").str.contains(firm.pv_regex))
    pids = m.select("patent_id").unique()
    joined = pids.join(pat, on="patent_id", how="inner")
    year_ct = (
        joined.group_by("year")
        .agg(pl.col("patent_id").n_unique().alias("n_patents"))
        .sort("year")
    )
    return year_ct, joined


def firm_top_cpc(patent_ids: pl.DataFrame, cpc: pl.DataFrame, n: int = 5) -> list[tuple[str, int]]:
    """Top-N CPC subclasses for a firm's patent set."""
    j = patent_ids.select("patent_id").unique().join(cpc, on="patent_id", how="inner")
    top = (
        j.group_by("cpc_subclass")
        .agg(pl.col("patent_id").n_unique().alias("n"))
        .sort("n", descending=True)
        .head(n)
    )
    return list(zip(top["cpc_subclass"].to_list(), top["n"].to_list()))


# ---------------------------------------------------------------------
# 2. Trademark + DeltaKL timeline (firm-year)
# ---------------------------------------------------------------------
def load_firm_trademarks(firm: Firm) -> pl.DataFrame:
    """Scan all 45 NICE classes, return all filings for the firm.

    Joins per-class surprise files to attach prospective_kl / retrospective_kl.
    """
    parts: list[pl.DataFrame] = []
    for cls in range(1, 46):
        tm_path = PROC / f"tm_class{cls:03d}.parquet"
        sp_path = PROC / f"surprise_class{cls:03d}.parquet"
        if not tm_path.exists():
            continue
        tm = pl.read_parquet(tm_path).filter(pl.col("owner_name").str.contains(firm.tm_regex))
        if tm.height == 0:
            continue
        tm = tm.with_columns(pl.lit(cls).cast(pl.Int32).alias("nice_class_int"))
        if sp_path.exists():
            sp = pl.read_parquet(sp_path).select(
                "serial_number", "prospective_kl", "retrospective_kl",
                "n_ref_prospective", "n_ref_retrospective"
            )
            tm = tm.join(sp, on="serial_number", how="left")
        else:
            tm = tm.with_columns(
                pl.lit(None).cast(pl.Float64).alias("prospective_kl"),
                pl.lit(None).cast(pl.Float64).alias("retrospective_kl"),
                pl.lit(None).cast(pl.Int32).alias("n_ref_prospective"),
                pl.lit(None).cast(pl.Int32).alias("n_ref_retrospective"),
            )
        parts.append(tm)
    if not parts:
        return pl.DataFrame()
    out = pl.concat(parts, how="diagonal_relaxed")
    out = out.with_columns(
        pl.col("filing_date").str.slice(0, 4).cast(pl.Int32, strict=False).alias("year"),
        (pl.col("prospective_kl") - pl.col("retrospective_kl")).alias("dkl"),
    )
    # Replace NaN with null so mean() skips them rather than propagating NaN
    out = out.with_columns(
        pl.when(pl.col("prospective_kl").is_nan()).then(None).otherwise(pl.col("prospective_kl")).alias("prospective_kl"),
        pl.when(pl.col("retrospective_kl").is_nan()).then(None).otherwise(pl.col("retrospective_kl")).alias("retrospective_kl"),
        pl.when(pl.col("dkl").is_nan()).then(None).otherwise(pl.col("dkl")).alias("dkl"),
    )
    return out


def firm_dkl_yearly(tm: pl.DataFrame) -> pl.DataFrame:
    """Mean DeltaKL per year (one row per distinct serial_number)."""
    if tm.is_empty():
        return pl.DataFrame()
    # Dedupe to one row per filing (a filing in two NICE classes shows up
    # twice; we collapse to one row per serial_number using the first
    # observation. The surprise file is computed per (serial_number,
    # class), and to keep DeltaKL well-defined we average across classes.)
    by_filing = (
        tm.group_by("serial_number")
        .agg(
            pl.col("year").first(),
            pl.col("filing_date").first(),
            pl.col("mark_identification").first(),
            pl.col("dkl").mean().alias("dkl"),
            pl.col("prospective_kl").mean().alias("prospective_kl"),
            pl.col("retrospective_kl").mean().alias("retrospective_kl"),
        )
    )
    yr = (
        by_filing.drop_nulls("year")
        .group_by("year")
        .agg(
            pl.col("serial_number").n_unique().alias("n_filings"),
            pl.col("dkl").mean().alias("mean_dkl"),
            pl.col("dkl").drop_nulls().count().alias("n_dkl"),
            pl.col("prospective_kl").mean().alias("mean_pros"),
            pl.col("retrospective_kl").mean().alias("mean_retr"),
        )
        .sort("year")
    )
    return yr


# ---------------------------------------------------------------------
# 3. Anchor-mark lookup
# ---------------------------------------------------------------------
def anchor_rows(tm: pl.DataFrame, firm: Firm) -> pl.DataFrame:
    rows = []
    if tm.is_empty():
        return pl.DataFrame()
    # For each anchor mark, find the earliest filing of the firm whose
    # mark_identification *equals* or *starts with* the anchor.
    by_filing = (
        tm.filter(pl.col("filing_date").is_not_null())
        .sort("filing_date")
        .group_by("serial_number")
        .agg(
            pl.col("filing_date").first(),
            pl.col("year").first(),
            pl.col("mark_identification").first(),
            pl.col("owner_name").first(),
            pl.col("goods_services").first(),
            pl.col("nice_classes").first(),
            pl.col("dkl").mean().alias("dkl"),
            pl.col("prospective_kl").mean().alias("prospective_kl"),
            pl.col("retrospective_kl").mean().alias("retrospective_kl"),
        )
        .sort("filing_date")
    )
    for brand, story, launch_yr in firm.anchors:
        # Match exact mark, "BRAND ..." prefix, or "... BRAND ..." anywhere
        # as a word boundary (catches "NVIDIA RTX", "DYSON SUPERSONIC" etc.)
        cand = by_filing.filter(
            (pl.col("mark_identification") == brand)
            | pl.col("mark_identification").str.contains(rf"(?i)\b{brand}\b")
        )
        if cand.height == 0:
            rows.append({
                "brand": brand,
                "story": story,
                "launch_year": launch_yr,
                "filing_date": None,
                "serial_number": None,
                "mark_identification": None,
                "nice_classes": None,
                "dkl": None,
                "prospective_kl": None,
                "retrospective_kl": None,
                "goods_services": None,
            })
            continue
        first = cand.row(0, named=True)
        rows.append({
            "brand": brand,
            "story": story,
            "launch_year": launch_yr,
            "filing_date": first["filing_date"],
            "serial_number": first["serial_number"],
            "mark_identification": first["mark_identification"],
            "nice_classes": ",".join(first["nice_classes"] or []),
            "dkl": first["dkl"],
            "prospective_kl": first["prospective_kl"],
            "retrospective_kl": first["retrospective_kl"],
            "goods_services": (first["goods_services"] or "")[:240],
        })
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------
# 4. Per-firm writeup
# ---------------------------------------------------------------------
def write_firm_text(
    firm: Firm,
    pt: pl.DataFrame,
    tm_yr: pl.DataFrame,
    anchors: pl.DataFrame,
    top_cpc: list[tuple[str, int]],
    out_path: Path,
) -> None:
    L: list[str] = []
    L.append(f"FIRM CASE STUDY: {firm.label}")
    L.append("=" * (16 + len(firm.label)))
    L.append("")
    if pt.is_empty():
        L.append("No patents matched.")
    else:
        n = pt["n_patents"].sum()
        ymin = pt["year"].min()
        ymax = pt["year"].max()
        L.append(f"Patents: {n:,} grants, {ymin}-{ymax}.")
        if top_cpc:
            cpc_str = ", ".join(f"{s}({n})" for s, n in top_cpc)
            L.append(f"Top CPC subclasses: {cpc_str}")
        # Show peak years
        top = pt.sort("n_patents", descending=True).head(5)
        L.append("Peak patent-grant years:")
        for r in top.iter_rows(named=True):
            L.append(f"  {r['year']}: {r['n_patents']:,} grants")
    L.append("")
    if tm_yr.is_empty():
        L.append("No trademark filings matched.")
    else:
        nf = tm_yr["n_filings"].sum()
        L.append(f"Trademark filings (post-1995, all NICE classes): {nf}")
        # Top mean_dkl years (filter n_filings >= 3 to avoid noise)
        top = (
            tm_yr.filter((pl.col("year") >= 1995) & (pl.col("n_filings") >= 3))
            .drop_nulls("mean_dkl")
            .sort("mean_dkl", descending=True)
            .head(5)
        )
        if top.height:
            L.append("Top mean-DeltaKL years (n_filings >= 3):")
            for r in top.iter_rows(named=True):
                L.append(
                    f"  {r['year']}: mean_dkl={r['mean_dkl']:+.3f}  n={r['n_filings']}"
                )
        bot = (
            tm_yr.filter((pl.col("year") >= 1995) & (pl.col("n_filings") >= 3))
            .drop_nulls("mean_dkl")
            .sort("mean_dkl")
            .head(5)
        )
        if bot.height:
            L.append("Bottom mean-DeltaKL years (n_filings >= 3):")
            for r in bot.iter_rows(named=True):
                L.append(
                    f"  {r['year']}: mean_dkl={r['mean_dkl']:+.3f}  n={r['n_filings']}"
                )
    L.append("")
    L.append("ANCHOR MARKS (hand-curated, mapped to product launches):")
    L.append("-" * 56)
    if anchors.is_empty():
        L.append("(no anchors)")
    else:
        for r in anchors.iter_rows(named=True):
            tag = r["brand"]
            if r["filing_date"] is None:
                L.append(f"  {tag:<14}  NOT FOUND   {r['story']}")
                continue
            dkl = r["dkl"]
            dkl_str = f"{dkl:+.3f}" if dkl is not None and not (isinstance(dkl, float) and np.isnan(dkl)) else "  n/a"
            L.append(
                f"  {tag:<14}  {r['filing_date']}  dKL={dkl_str}  NICE={r['nice_classes']}  "
                f"[{r['story']}]"
            )
            gs = r["goods_services"]
            if gs:
                L.append(f"     g/s: {gs[:160]}")
    L.append("")
    L.append("YEARLY TIMELINE (year, patents, n_filings, mean_dkl):")
    L.append("-" * 56)
    if pt.is_empty() and tm_yr.is_empty():
        L.append("(nothing)")
    else:
        years = set()
        if not pt.is_empty():
            years |= set(pt["year"].to_list())
        if not tm_yr.is_empty():
            years |= set(tm_yr["year"].to_list())
        years_l = sorted(y for y in years if y is not None and y >= 1990)
        pt_d = {r["year"]: r["n_patents"] for r in pt.iter_rows(named=True)} if not pt.is_empty() else {}
        tm_d = {r["year"]: r for r in tm_yr.iter_rows(named=True)} if not tm_yr.is_empty() else {}
        for y in years_l:
            np_ = pt_d.get(y, 0)
            t = tm_d.get(y, None)
            if t is None:
                L.append(f"  {y}  pat={np_:>5}  tm=  0  mean_dkl=    n/a")
            else:
                md = t["mean_dkl"]
                md_str = f"{md:+.3f}" if md is not None and not (isinstance(md, float) and np.isnan(md)) else "    n/a"
                L.append(f"  {y}  pat={np_:>5}  tm={t['n_filings']:>3}  mean_dkl={md_str}")
    out_path.write_text("\n".join(L) + "\n")


# ---------------------------------------------------------------------
# 5. Multi-panel figure
# ---------------------------------------------------------------------
def make_figure(per_firm: dict[str, dict], out_path: Path) -> None:
    n = len(per_firm)
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 3.0 * nrows), sharex=True)
    axes = axes.flatten() if n > 1 else [axes]
    for i, (key, d) in enumerate(per_firm.items()):
        ax = axes[i]
        firm: Firm = d["firm"]
        pt: pl.DataFrame = d["patent_year"]
        tmy: pl.DataFrame = d["tm_year"]
        anchors: pl.DataFrame = d["anchors"]
        # Left axis: patent grant count (bars)
        ax.set_title(firm.label, fontsize=11)
        ax.set_xlim(1995, 2026)
        if not pt.is_empty():
            yrs = pt.filter(pl.col("year") >= 1990)["year"].to_list()
            cts = pt.filter(pl.col("year") >= 1990)["n_patents"].to_list()
            ax.bar(yrs, cts, color="0.78", width=0.85, label="patents (grants/yr)")
        ax.set_ylabel("patents/yr", fontsize=8, color="0.4")
        ax.tick_params(axis="y", labelsize=7, colors="0.4")
        # Reserve top quarter of the axis for anchor labels
        if not pt.is_empty() and ax.get_ylim()[1] > 0:
            ax.set_ylim(0, ax.get_ylim()[1] * 1.55)
        # Right axis: mean DeltaKL trajectory (line)
        ax2 = ax.twinx()
        if not tmy.is_empty():
            sub = tmy.filter((pl.col("year") >= 1995) & (pl.col("n_filings") >= 2))
            yrs = sub["year"].to_list()
            mds = [m if m is not None and not (isinstance(m, float) and np.isnan(m)) else np.nan
                   for m in sub["mean_dkl"].to_list()]
            ax2.plot(yrs, mds, "-", color="C3", lw=1.5, label="mean dKL")
            ax2.axhline(0.0, color="0.3", lw=0.4, linestyle="--")
        ax2.set_ylabel("mean dKL", fontsize=8, color="C3")
        ax2.tick_params(axis="y", labelsize=7, colors="C3")
        # Anchor annotations: vertical lines at filing year + rotated brand label
        if not anchors.is_empty():
            placed_years: dict[int, int] = {}
            for r in anchors.iter_rows(named=True):
                fd = r["filing_date"]
                if not fd:
                    continue
                try:
                    y = int(fd[:4])
                except (TypeError, ValueError):
                    continue
                if y < 1995 or y > 2026:
                    continue
                stack = placed_years.get(y, 0)
                placed_years[y] = stack + 1
                ax.axvline(y, color="C0", lw=0.5, alpha=0.45)
                ymax = ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1
                # Rotate vertical, anchor at the top of the axis
                ax.text(
                    y, ymax * (0.98 - 0.10 * (stack % 3)),
                    r["brand"],
                    fontsize=6.5, color="C0", rotation=90,
                    ha="center", va="top",
                )
    # Hide spare axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    axes[-2].set_xlabel("year")
    axes[-1].set_xlabel("year")
    fig.suptitle(
        "Patent grants vs trademark dKL by firm, with anchor-mark filings",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> int:
    print("[1/4] Loading PatentsView (this takes ~30s) …", file=sys.stderr)
    ag, pat = load_patent_data()

    print("[1b/4] Loading g_cpc_current.tsv …", file=sys.stderr)
    cpc = pl.scan_csv(
        PV_DIR / "g_cpc_current.tsv", separator="\t",
        schema_overrides={"patent_id": pl.Utf8},
        ignore_errors=True,
    ).select("patent_id", "cpc_subclass").filter(
        pl.col("cpc_subclass").is_not_null()
    ).unique().collect()
    print(f"      {cpc.height:,} (patent, cpc_subclass) rows", file=sys.stderr)

    per_firm: dict[str, dict] = {}
    summary_rows: list[dict] = []

    print("[2/4] Per-firm aggregation …", file=sys.stderr)
    for firm in FIRMS:
        print(f"  - {firm.label}", file=sys.stderr)
        pt_yr, pt_full = firm_patent_timeline(ag, pat, firm)
        top_cpc = firm_top_cpc(pt_full, cpc, n=5) if not pt_full.is_empty() else []
        tm = load_firm_trademarks(firm)
        tm_yr = firm_dkl_yearly(tm)
        anchors = anchor_rows(tm, firm)

        # Write per-firm text + CSV
        out_txt = RESULTS / f"case_study_{firm.key}.txt"
        write_firm_text(firm, pt_yr, tm_yr, anchors, top_cpc, out_txt)
        # Combined timeline CSV
        if not pt_yr.is_empty() or not tm_yr.is_empty():
            years = sorted(set((pt_yr["year"].to_list() if not pt_yr.is_empty() else []) +
                               (tm_yr["year"].to_list() if not tm_yr.is_empty() else [])))
            years = [y for y in years if y is not None and y >= 1990]
            pt_d = {r["year"]: r["n_patents"] for r in pt_yr.iter_rows(named=True)} if not pt_yr.is_empty() else {}
            tm_d = {r["year"]: r for r in tm_yr.iter_rows(named=True)} if not tm_yr.is_empty() else {}
            tl = pl.DataFrame({
                "year": years,
                "n_patents": [pt_d.get(y, 0) for y in years],
                "n_filings": [(tm_d.get(y) or {}).get("n_filings", 0) for y in years],
                "mean_dkl":  [(tm_d.get(y) or {}).get("mean_dkl", None) for y in years],
                "mean_pros": [(tm_d.get(y) or {}).get("mean_pros", None) for y in years],
                "mean_retr": [(tm_d.get(y) or {}).get("mean_retr", None) for y in years],
            })
            tl.write_csv(RESULTS / f"case_study_{firm.key}_timeline.csv")
        if not anchors.is_empty():
            anchors.write_csv(RESULTS / f"case_study_{firm.key}_anchors.csv")

        per_firm[firm.key] = {
            "firm": firm,
            "patent_year": pt_yr,
            "tm_year": tm_yr,
            "anchors": anchors,
            "top_cpc": top_cpc,
        }

        n_pat = int(pt_yr["n_patents"].sum()) if not pt_yr.is_empty() else 0
        n_tm  = int(tm_yr["n_filings"].sum()) if not tm_yr.is_empty() else 0
        anchor_hits = int(anchors.filter(pl.col("filing_date").is_not_null()).height) if not anchors.is_empty() else 0
        summary_rows.append({
            "firm": firm.label,
            "key": firm.key,
            "n_patents": n_pat,
            "n_tm_filings": n_tm,
            "n_anchors_attempted": len(firm.anchors),
            "n_anchors_found": anchor_hits,
            "top_cpc": ";".join(s for s, _ in top_cpc),
        })

    # Free memory before figure
    del ag, pat, cpc
    gc.collect()

    print("[3/4] Writing summary CSV …", file=sys.stderr)
    pl.DataFrame(summary_rows).write_csv(RESULTS / "firm_case_studies_summary.csv")

    print("[4/4] Writing figure …", file=sys.stderr)
    make_figure(per_firm, RESULTS / "firm_case_studies.png")
    print("done.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
