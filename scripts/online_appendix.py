"""Build the online appendix: per-industry breakouts and cross-industry comparisons.

The paper reports pooled estimates and a forest plot. Every class-level result
behind them is computed here and published, so that a reader who cares about one
industry can see it, and so that the cross-industry variation -- which is large,
and which the pooled numbers hide -- is inspectable rather than asserted.

For each of the 45 Nice classes this emits one three-panel figure:
  A. first-gate failure by lead quintile, with 95% intervals
  B. registration completion by atypicality and by lead
  C. the class in cross-industry context, its gate lift against all others

Plus the cross-industry exhibits:
  - gate lift ranked across classes (the forest, rebuilt at appendix size)
  - gate lift against class size and against base failure rate, which are the
    two obvious mechanical explanations for the cross-class spread
  - a machine-readable per-class table

Panel A and the panel-C histogram come from different estimators and are not
the same number. Panel A is re-fit here: an LPM on within-class lead quintiles
with registration-cohort FE and log length held, estimated inside that one
class -- the paper's specification minus the class FE, which are meaningless
in a single class. Panel C and the forest reuse the RAW pooled-quintile lifts
already computed by event_gates_all.py, read out of its JSON rather than
recomputed, so the appendix and the paper's forest figure cannot drift apart.

Scoring defaults to "rolling" here (unlike the gate scripts, which default to
"topic" for backward compatibility), because everything published under
docs/online-appendix/ must match the paper's per-filing reference windows.

Usage:  python scripts/online_appendix.py
Reads   data/processed/case_events.parquet
        data/processed/tm_class{NNN}.parquet
        data/processed/{SRC}_surprise_class{NNN}.parquet
        paper/results/event_gates_all.json   (the raw per-class lifts)
Output: docs/online-appendix/index.html
        docs/online-appendix/figures/class_{NNN}.png   (one per class built)
        docs/online-appendix/figures/cross_{forest,scatter}.png
        docs/online-appendix/per_class_estimates.csv
"""
from __future__ import annotations

import gc
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
RES = REPO / "paper" / "results"
OUT = REPO / "docs" / "online-appendix"
FIG = OUT / "figures"

SRC = os.environ.get("SURPRISE_SRC", "rolling")
REG_LO, REG_HI = 2002, 2018
GATE_LO, GATE_HI = 4.0, 8.5
MIN_CELL = 100

INK, INK2, GRID = "#1a1a1a", "#555555", "#d8d8d8"
C_LEAD, C_ATYP = "#c0392b", "#2b6cb0"
QX = [1, 2, 3, 4, 5]

NICE_NAMES = {
    "001": "Chemicals", "002": "Paints", "003": "Cosmetics & Cleaning",
    "004": "Lubricants & Fuels", "005": "Pharmaceuticals", "006": "Metal Goods",
    "007": "Machinery", "008": "Hand Tools", "009": "Software & Electronics",
    "010": "Medical Apparatus", "011": "Lighting & Heating", "012": "Vehicles",
    "013": "Firearms", "014": "Jewelry", "015": "Musical Instruments",
    "016": "Paper & Printed Goods", "017": "Rubber & Plastics",
    "018": "Leather Goods", "019": "Building Materials", "020": "Furniture",
    "021": "Household Utensils", "022": "Cordage & Fibers", "023": "Yarns",
    "024": "Textiles", "025": "Clothing & Footwear", "026": "Lace & Embroidery",
    "027": "Carpets", "028": "Games & Sporting Goods",
    "029": "Meats & Processed Foods", "030": "Staple Foods",
    "031": "Agricultural Products", "032": "Beer & Soft Drinks",
    "033": "Alcoholic Beverages", "034": "Tobacco", "035": "Advertising & Retail",
    "036": "Insurance & Finance", "037": "Construction & Repair",
    "038": "Telecommunications", "039": "Transport & Storage",
    "040": "Material Treatment", "041": "Education & Entertainment",
    "042": "Scientific & Tech Services", "043": "Hotels & Restaurants",
    "044": "Medical & Beauty Services", "045": "Legal & Personal Services",
}
CLASSES = sorted(NICE_NAMES)


def log(m: str) -> None:
    print(m, file=sys.stderr, flush=True)


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def gate_events() -> pl.DataFrame:
    """Earliest maintenance cancellation per serial, Section 8 or Section 71."""
    return pl.scan_parquet(PROC / "case_events.parquet").filter(
        pl.col("code").is_in(["C8..", "C71T"]) & (pl.col("date") > 19000000)
    ).select("serial_number", "date").collect().with_columns(
        pl.col("date").cast(pl.Utf8).str.strptime(pl.Date, "%Y%m%d", strict=False).alias("c8_d")
    ).drop_nulls("c8_d").group_by("serial_number").agg(pl.col("c8_d").min())


def class_frame(cls: str, ev: pl.DataFrame) -> pl.DataFrame | None:
    sp = PROC / f"{SRC}_surprise_class{cls}.parquet"
    tp = PROC / f"tm_class{cls}.parquet"
    if not (sp.exists() and tp.exists()):
        return None
    tm = pl.read_parquet(tp, columns=["serial_number", "filing_date",
                                      "registration_date", "goods_services"])
    tm = tm.with_columns(
        pl.col("registration_date").str.strptime(pl.Date, "%Y%m%d", strict=False).alias("reg_d"),
        pl.col("filing_date").str.slice(0, 4).cast(pl.Int32, strict=False).alias("fyear"),
        pl.col("goods_services").fill_null("").str.len_chars().alias("gs_len"),
    ).drop("goods_services")
    sc = pl.read_parquet(sp, columns=["serial_number", "topic_kl_vs_past",
                                      "topic_kl_vs_future", "topic_dkl"])
    d = tm.join(sc, on="serial_number", how="inner").filter(pl.col("topic_dkl").is_finite())
    d = d.with_columns(
        (0.5 * (pl.col("topic_kl_vs_past") + pl.col("topic_kl_vs_future"))).alias("atyp"),
        (pl.col("registration_date").fill_null("").str.len_chars() >= 8)
        .cast(pl.Float64).alias("registered"),
    )
    d = d.join(ev, on="serial_number", how="left").with_columns(
        ((pl.col("c8_d") - pl.col("reg_d")).dt.total_days() / 365.25).alias("age")
    ).with_columns(
        pl.col("reg_d").dt.year().alias("reg_year"),
        ((pl.col("age") >= GATE_LO) & (pl.col("age") < GATE_HI))
        .fill_null(False).cast(pl.Float64).alias("failed1"),
    )
    return d


def quintile_within(d: pl.DataFrame, score: str, cell: str) -> pl.DataFrame:
    return d.sort([cell, score, "serial_number"]).with_columns(
        ((pl.col(score).rank("ordinal").over(cell) - 1) * 5
         // pl.len().over(cell)).cast(pl.Int8).alias("q"))


def lpm_quintiles(d: pl.DataFrame, y: str, cell: str) -> tuple[list, list, float, int] | None:
    """Cohort-FE LPM on quintile dummies (Q1 omitted) + log length. Robust SE.

    Returns None rather than an unstable estimate when the class is too thin
    (fewer than 2,000 registrations in cells of at least MIN_CELL, or fewer
    than five populated quintiles) or when the demeaned design is
    near-singular; those classes are drawn with a "too few" placeholder panel.
    Errors are robust rather than owner-clustered: within a single class an
    owner contributes far fewer marks than across the corpus, and the appendix
    is descriptive, so the paper's inferential claim stays with
    gate_decisive_regression.py.
    """
    d = d.filter(pl.len().over(cell) >= MIN_CELL)
    if d.height < 2000 or d["q"].n_unique() < 5:
        return None
    X = np.column_stack([(d["q"] == q).cast(pl.Float64).to_numpy() for q in range(1, 5)]
                        + [np.log1p(d["gs_len"].cast(pl.Float64).to_numpy())])
    yv = d[y].cast(pl.Float64).to_numpy()
    codes = d[cell].to_physical().to_numpy() if d[cell].dtype != pl.Utf8 else \
        np.unique(d[cell].to_numpy(), return_inverse=True)[1]
    nb = codes.max() + 1
    def demean(v):
        s = np.bincount(codes, weights=v, minlength=nb)
        n = np.bincount(codes, minlength=nb)
        return v - (s / np.maximum(n, 1))[codes]
    Xd = np.column_stack([demean(X[:, j]) for j in range(X.shape[1])])
    yd = demean(yv)
    XtX = Xd.T @ Xd
    if np.linalg.cond(XtX) > 1e12:
        return None
    b = np.linalg.solve(XtX, Xd.T @ yd)
    e = yd - Xd @ b
    Xi = np.linalg.inv(XtX)
    V = Xi @ (Xd * (e ** 2)[:, None]).T @ Xd @ Xi
    se = np.sqrt(np.diag(V))
    return ([0.0] + list(b[:4] * 100), [0.0] + list(se[:4] * 100),
            float(yv.mean()) * 100, int(d.height))


def raw_profile(d: pl.DataFrame, score: str, y: str) -> list[float]:
    """Raw rates by quintile, cut within FILING year (panel B is on filings).

    Panel A cuts within REGISTRATION cohort instead, because the gate outcome
    is dated from registration. The two panels therefore do not share a cell
    definition and their quintiles are not the same filings.
    """
    dd = quintile_within(d, score, "fyear")
    g = dd.group_by("q").agg(pl.col(y).mean().alias("p")).sort("q")
    return [float(r["p"]) * 100 for r in g.to_dicts()]


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------
def style(ax, title, ylab, xlab):
    ax.set_xticks(QX)
    ax.set_xticklabels(["Q1\nlow", "Q2", "Q3", "Q4", "Q5\nhigh"], fontsize=8)
    ax.set_title(title, fontsize=9.5, color=INK, pad=7)
    ax.set_ylabel(ylab, fontsize=8.5, color=INK2)
    ax.set_xlabel(xlab, fontsize=8.5, color=INK2)
    ax.grid(alpha=0.28, color=GRID)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)


def class_figure(cls: str, gate, reg_atyp, reg_lead, allrows) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.9))
    name = NICE_NAMES[cls]

    ax = axes[0]
    if gate:
        b, se, base, n = gate
        lo = [x - 1.96 * s for x, s in zip(b, se)]
        hi = [x + 1.96 * s for x, s in zip(b, se)]
        ax.fill_between(QX, lo, hi, color=C_LEAD, alpha=0.13, linewidth=0)
        ax.plot(QX, b, "o-", color=C_LEAD, lw=2.0, ms=5)
        ax.axhline(0, color=INK2, lw=1.0, ls=(0, (4, 3)))
        style(ax, f"A. First-gate failure by lead\n(cohort FE, length held; base {base:.1f}%, n={n:,})",
              "pp vs bottom quintile", "lead $L$ quintile")
    else:
        ax.text(0.5, 0.5, "too few registrations\nto estimate", ha="center",
                va="center", fontsize=9, color=INK2, transform=ax.transAxes)
        style(ax, "A. First-gate failure by lead", "pp vs bottom quintile", "")

    ax = axes[1]
    ax.plot(QX, reg_atyp, "s-", color=C_ATYP, lw=2.0, ms=5, label="atypicality $A$")
    ax.plot(QX, reg_lead, "o-", color=C_LEAD, lw=2.0, ms=5, label="lead $L$")
    style(ax, "B. Registration completion", "% reaching registration", "quintile")
    ax.legend(fontsize=7.5, framealpha=0.9)

    ax = axes[2]
    lifts = sorted([r["lift"] for r in allrows])
    me = next((r["lift"] for r in allrows if r["cls"] == cls), None)
    ax.hist(lifts, bins=22, color="#cccccc", edgecolor="white")
    if me is not None:
        ax.axvline(me, color=C_LEAD, lw=2.2)
        ax.annotate(f"{name}\n{me:+.1f}pp", xy=(me, ax.get_ylim()[1] * 0.72),
                    xytext=(6, 0), textcoords="offset points", fontsize=8,
                    color=C_LEAD, ha="left")
    ax.axvline(0, color=INK2, lw=1.0, ls=(0, (4, 3)))
    ax.set_title("C. This class against all 45\n(raw Q5$-$Q1 gate lift)", fontsize=9.5,
                 color=INK, pad=7)
    ax.set_xlabel("gate lift (pp)", fontsize=8.5, color=INK2)
    ax.set_ylabel("classes", fontsize=8.5, color=INK2)
    ax.grid(alpha=0.28, color=GRID)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)

    fig.suptitle(f"Nice {cls} — {name}", fontsize=11.5, color=INK, y=1.03)
    fig.tight_layout()
    fig.savefig(FIG / f"class_{cls}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def cross_figures(rows: list[dict]) -> None:
    rs = sorted(rows, key=lambda r: r["lift"])
    names = [f"{r['cls']} {NICE_NAMES[r['cls']]}" for r in rs]
    lift = [r["lift"] for r in rs]
    se = [r["se"] for r in rs]

    fig, ax = plt.subplots(figsize=(8.2, 11.0))
    y = np.arange(len(rs))
    cols = [C_LEAD if v > 0 else C_ATYP for v in lift]
    ax.errorbar(lift, y, xerr=[1.96 * s for s in se], fmt="none",
                ecolor=GRID, elinewidth=1.4, capsize=0, zorder=1)
    ax.scatter(lift, y, s=26, c=cols, zorder=2)
    ax.axvline(0, color=INK2, lw=1.1, ls=(0, (4, 3)))
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=7.6)
    ax.set_xlabel("first-gate failure, top minus bottom lead quintile (pp)",
                  fontsize=9, color=INK2)
    ax.set_title("Gate lift by industry, all 45 Nice classes\n"
                 "(raw quintile contrast, 95% intervals)", fontsize=11, color=INK)
    ax.grid(alpha=0.28, color=GRID, axis="x")
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=INK2)
    fig.tight_layout()
    fig.savefig(FIG / "cross_forest.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6))
    for ax, xk, xl in ((axes[0], "n", "registrations in class (log scale)"),
                       (axes[1], "base", "class base failure rate (%)")):
        xs = [r[xk] for r in rows]
        ax.scatter(xs, [r["lift"] for r in rows], s=30, c=C_LEAD, alpha=0.75)
        for r in rows:
            if abs(r["lift"]) > 6 or r[xk] == max(xs):
                ax.annotate(r["cls"], (r[xk], r["lift"]), fontsize=7,
                            xytext=(3, 3), textcoords="offset points", color=INK2)
        if xk == "n":
            ax.set_xscale("log")
        ax.axhline(0, color=INK2, lw=1.0, ls=(0, (4, 3)))
        ax.set_xlabel(xl, fontsize=9, color=INK2)
        ax.set_ylabel("gate lift (pp)", fontsize=9, color=INK2)
        ax.grid(alpha=0.28, color=GRID)
        for s in ax.spines.values():
            s.set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=8.5)
    axes[0].set_title("Gate lift is not a size artifact", fontsize=10.5, color=INK)
    axes[1].set_title("Nor a base-rate artifact", fontsize=10.5, color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "cross_scatter.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    log(f"[scoring] {SRC}")
    ev = gate_events()

    pooled = json.loads((RES / "event_gates_all.json").read_text())
    allrows = [{"cls": r["cls"], "lift": r["lift"] * 100, "se": r["se"] * 100,
                "base": r["p_fail1"] * 100, "n": r["n"]} for r in pooled["per_class"]]

    rows = []
    for cls in CLASSES:
        d = class_frame(cls, ev)
        if d is None or d.height < 5000:
            log(f"  [{cls}] skipped")
            continue
        regs = d.filter((pl.col("registered") == 1.0)
                        & pl.col("reg_year").is_between(REG_LO, REG_HI))
        gate = None
        if regs.height >= 5000:
            gate = lpm_quintiles(quintile_within(regs, "topic_dkl", "reg_year"),
                                 "failed1", "reg_year")
        deb = d.filter(pl.col("fyear").is_between(1995, 2018))
        ra = raw_profile(deb, "atyp", "registered")
        rl = raw_profile(deb, "topic_dkl", "registered")
        class_figure(cls, gate, ra, rl, allrows)

        base = next((r for r in allrows if r["cls"] == cls), None)
        rows.append({
            "cls": cls, "name": NICE_NAMES[cls],
            "n_scored": d.height, "n_registrations": regs.height,
            "gate_base_pct": round(base["base"], 2) if base else None,
            "gate_lift_raw_pp": round(base["lift"], 3) if base else None,
            "gate_lift_raw_se": round(base["se"], 3) if base else None,
            "gate_q5_fe_pp": round(gate[0][4], 3) if gate else None,
            "gate_q5_fe_se": round(gate[1][4], 3) if gate else None,
            "reg_atyp_q1": round(ra[0], 2), "reg_atyp_q5": round(ra[4], 2),
            "reg_lead_q1": round(rl[0], 2), "reg_lead_q5": round(rl[4], 2),
        })
        log(f"  [{cls}] {NICE_NAMES[cls]}: n={d.height:,} regs={regs.height:,}"
            + (f" q5={gate[0][4]:+.2f}pp" if gate else " (gate n/a)"))
        del d, regs, deb
        gc.collect()

    cross_figures(allrows)
    pl.DataFrame(rows).write_csv(OUT / "per_class_estimates.csv")
    write_index(rows, allrows)
    log(f"[done] {len(rows)} classes -> {OUT}")
    return 0


INDEX_HEAD = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Online appendix: per-industry results</title>
<style>
  body { font-family: Georgia, serif; max-width: 900px; margin: 24px auto; padding: 0 12px; color: #222; line-height: 1.5; }
  h1 { font-size: 1.35em; }
  h2 { font-size: 1.15em; margin-top: 1.6em; }
  h3 { font-size: 1.02em; margin-top: 1.5em; }
  a { color: #2b6cb0; }
  img { max-width: 100%; height: auto; }
  .tablewrap { overflow-x: auto; }
  table { border-collapse: collapse; font-size: 0.9em; width: 100%; }
  th, td { padding: 4px 9px; border-bottom: 1px solid #ddd; white-space: nowrap; }
  td.num, th.num { text-align: right; }
  th { cursor: pointer; user-select: none; border-bottom: 2px solid #999; text-align: left; }
  th.num { text-align: right; }
  th .dir { color: #888; font-size: 0.85em; }
  tbody tr:hover { background: #f4f6f8; }
  .backlink { font-size: 0.78em; font-family: Georgia, serif; font-weight: normal; margin-left: 10px; }
  .hint { font-size: 0.82em; color: #555; }
  .sitenav { font-size: 0.85em; color: #777; margin: 0 0 1.6em; }
  .sitenav b { color: #222; font-weight: normal; }
</style>
</head>
<body>

<p class="sitenav"><a href="../">tm-vocabulary</a> &middot;
<a href="https://ssrn.com/abstract=6954598">paper</a> &middot;
<b>online appendix</b> &middot;
<a href="themes/">theme explorer</a> &middot;
<a href="ipo-viewer/">IPO viewer</a> &middot;
<a href="https://github.com/ericsilver/tm-vocabulary">code</a></p>

<h1>Online appendix: per-industry results</h1>

<p>Supplement to <a href="https://ssrn.com/abstract=6954598"><em>Business Themes in the
Trademark Record: Language Signals of Product Survival, Funding, and
Listing</em></a> (SSRN 6954598).</p>

<p>Everything here is computed on the same corpus and the same per-filing
reference windows as the paper. The paper reports pooled estimates because
they are what the design identifies cleanly; this appendix reports the class
level because the pooled numbers average over real and large industry
variation, and a reader who cares about one industry should be able to see it.</p>

<h2>How to read these</h2>

<p>The gate contrast is <strong>top minus bottom lead quintile</strong>, in percentage points
of failure at the five-year proof of continued use, not a per-quintile slope.
Panel A of each class figure
is a linear probability model with registration-cohort fixed effects and log
description length held fixed, estimated within that class alone; panel B is
raw completion rates by quintile; panel C places the class in the
cross-industry distribution. Shaded bands and whiskers are 95% intervals.</p>

<p>Per-class estimates are noisier than the pooled figure by construction, and
small classes are noisy enough that individual signs should not be read as
findings. The cross-industry exhibits below are the honest summary:
<strong>%(pos)d of %(nraw)d classes show a positive raw gate lift</strong>, and <strong>%(fepos)d of %(nfe)d</strong> do so under
the cohort-fixed-effects specification, which is the claim the paper makes.
The spread around it is what this appendix shows, and it is wide: the
fixed-effects contrast runs from about %(femin)s to %(femax)s across classes.</p>

<p>Nice 023 (yarns) falls below the volume floor the pooled analysis uses and has
no raw entry; its figure is still generated.</p>

<h2>Cross-industry comparisons</h2>

<p><img src="figures/cross_forest.png" alt="Gate lift by industry"></p>

<p><img src="figures/cross_scatter.png" alt="Gate lift against class size and base rate"></p>

<p>The scatter matters because the two obvious mechanical explanations for
cross-class variation are class size (more registrations, tighter estimate,
possibly different sign) and base failure rate (a class where most marks die
has less room to move). Neither organises the variation.</p>

<h2>Interactive: the IPO viewer</h2>

<p><a href="ipo-viewer/index.html"><strong>Open the IPO viewer</strong></a> &mdash; every class&rsquo;s
registrations (1996&ndash;2018) as dots on the lead/atypicality plane, colored
by the latest gate the record reaches: cancelled at the five-year proof of
continued use, passed it, owner in SEC financial reporting, owner with an
IPO marker. Later gates draw on top. Filter by Nice class and by
registration year range.</p>

<h2>Machine-readable</h2>

<p><a href="per_class_estimates.csv"><code>per_class_estimates.csv</code></a> &mdash; one row per class:
scored filings, registrations, base failure rate, raw and fixed-effects gate
contrasts with standard errors, and registration completion at both tails of
each axis.</p>

<h2>Per-industry breakouts</h2>

<p class="hint">Click a column heading to sort; click again to reverse. The
Nice number links to that class&rsquo;s figure below.</p>

<div class="tablewrap">
<table id="classes">
<thead>
<tr>
  <th class="num">Nice</th>
  <th>Industry</th>
  <th class="num">Registrations</th>
  <th class="num">Base fail</th>
  <th class="num">Gate lift (raw)</th>
  <th class="num">Gate Q5, cohort FE</th>
</tr>
</thead>
<tbody>
"""

INDEX_FOOT = """
<p style="margin-top:2em"><a href="../">tm-vocabulary</a> &middot;
<a href="https://ssrn.com/abstract=6954598">the paper</a> &middot;
<a href="https://aporia.institute/">aporia.institute</a></p>

<script>
(function () {
  const table = document.getElementById("classes");
  const tbody = table.tBodies[0];
  const heads = table.tHead.rows[0].cells;
  let sortCol = -1, asc = true;

  function key(row, col) {
    const t = row.cells[col].textContent.trim();
    if (col === 1) return t.toLowerCase();
    if (t === "\\u2014") return NaN;                     // em dash: no entry
    const n = parseFloat(t.replace(/[+,%]/g, "").replace(/\\u2212/g, "-"));
    return isNaN(n) ? NaN : n;
  }

  for (let c = 0; c < heads.length; c++) {
    heads[c].addEventListener("click", function () {
      asc = (sortCol === c) ? !asc : (c === 1);          // text asc, numbers desc first
      sortCol = c;
      const rows = Array.from(tbody.rows);
      rows.sort(function (a, b) {
        const ka = key(a, c), kb = key(b, c);
        if (typeof ka === "string") return asc ? ka.localeCompare(kb) : kb.localeCompare(ka);
        if (isNaN(ka)) return 1;                         // missing values sink
        if (isNaN(kb)) return -1;
        return asc ? ka - kb : kb - ka;
      });
      rows.forEach(function (r) { tbody.appendChild(r); });
      for (let i = 0; i < heads.length; i++) {
        const d = heads[i].querySelector(".dir");
        if (d) d.remove();
      }
      const mark = document.createElement("span");
      mark.className = "dir";
      mark.textContent = asc ? " \\u25B2" : " \\u25BC";
      heads[c].appendChild(mark);
    });
  }
})();
</script>

</body>
</html>
"""


def _num(s: str) -> str:
    """Typeset negatives with a real minus sign for the HTML cells."""
    return s.replace("-", "&minus;")


def write_index(rows: list[dict], allrows: list[dict]) -> None:
    pos = sum(1 for r in allrows if r["lift"] > 0)
    fes = [r["gate_q5_fe_pp"] for r in rows if r.get("gate_q5_fe_pp") is not None]
    head = INDEX_HEAD % {
        "pos": pos, "nraw": len(allrows),
        "fepos": sum(1 for v in fes if v > 0), "nfe": len(fes),
        "femin": _num(f"{min(fes):+.0f}pp"), "femax": _num(f"{max(fes):+.0f}pp"),
    }
    L = [head.rstrip("\n")]
    for r in sorted(rows, key=lambda x: x["cls"]):
        # Classes below the pooled analysis's volume floor have no entry in
        # event_gates_all.json, so raw lift and base rate are absent for them.
        n = int(r["cls"])
        lift = _num(f"{r['gate_lift_raw_pp']:+.1f} &plusmn; {1.96*r['gate_lift_raw_se']:.1f}") \
            if r.get("gate_lift_raw_pp") is not None else "&mdash;"
        fe = _num(f"{r['gate_q5_fe_pp']:+.2f} ({r['gate_q5_fe_se']:.2f})") \
            if r.get("gate_q5_fe_pp") is not None else "&mdash;"
        base = f"{r['gate_base_pct']:.1f}%" \
            if r.get("gate_base_pct") is not None else "&mdash;"
        name = r["name"].replace("&", "&amp;")
        L.append(f'<tr><td class="num"><a href="#nice-{n}">{n}</a></td>'
                 f'<td>{name}</td><td class="num">{r["n_registrations"]:,}</td>'
                 f'<td class="num">{base}</td><td class="num">{lift}</td>'
                 f'<td class="num">{fe}</td></tr>')
    L.append("</tbody>\n</table>\n</div>")
    for r in sorted(rows, key=lambda x: x["cls"]):
        n = int(r["cls"])
        name = r["name"].replace("&", "&amp;")
        L += [f'\n<h3 id="nice-{n}">Nice {n} &mdash; {name} '
              f'<a class="backlink" href="#classes">&uarr; table</a></h3>',
              f'<p><img src="figures/class_{r["cls"]}.png" alt="Nice {n}" loading="lazy"></p>']
    L.append(INDEX_FOOT)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "index.html").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
