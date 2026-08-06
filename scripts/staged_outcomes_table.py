"""One table: every outcome in the trademark funnel, conditioned on its prerequisite.

The paper reports its outcomes in separate places on separate samples, which
makes it hard to see the thing that actually matters -- that the two axes of
vocabulary position do different work at different stages, and that the stage
where each one bites is informative. This script runs the SAME specification on
every outcome, each conditioned on having reached the prior stage, so the
coefficients are read down a single column.

Unit of observation is the DEBUT OWNER (the owner's first-ever filing), scored
on that filing, so one firm contributes one row and the funnel is a funnel.

Specification, identical at every stage:
    outcome ~ z_level + z_lean + log_len,  class x debut-year fixed effects,
    heteroskedasticity-robust standard errors
where
    z_level = standardized A = (KL vs past + KL vs future)/2  ("atypicality")
    z_lean  = standardized dKL = KL(vs past) - KL(vs future)  ("lead")
Coefficients are in percentage points of the outcome.

Stages:
    1. registered    | filed            -- did the debut filing complete registration
    2. passed_gate   | registered       -- did it survive the first use-proof gate
    3. funded        | filed            -- did the owner ever file a Reg D notice
    4. funded        | registered       -- same, among owners who registered
    5. listed        | funded           -- did a funded owner reach the listed universe
    6. listed        | registered       -- the paper's published unconditional comparison

The gate is the first maintenance filing, which is Section 8 for domestic
registrations and Section 71 for Madrid Section 66(a) extensions of protection;
both cancellation codes are read (see gate_outcome).

Stage windows differ and are reported per row: the gate needs elapsed
time, and Form D is only electronic from 2009, so no single cohort supports
every stage. That is a property of the institutions, not a defect of the design,
but it means rows are not a strict nested decomposition and the table says so.

This table alone is estimated at T = 200 topics rather than the T = 50 used
elsewhere in the paper; the sign and ordering of every row are unchanged at
T = 50.

Inputs : data/processed/tm_class{cls}.parquet
         data/processed/{SRC}_surprise_class{cls}_T200.parquet
             SRC = SURPRISE_SRC, "rolling" for the per-filing windows the paper
             uses, "topic" for the retired per-calendar-year buckets
         data/processed/case_events.parquet
         data/processed/uspto_sec_crosswalk.parquet
         data/processed/funding_owner_match.parquet  (written by the funding
             script; if absent the three financing rows are skipped rather
             than the run failing)
Output : paper/results/staged_outcomes.json
         paper/results/staged_outcomes_table.tex   (input directly by the paper)
"""
from __future__ import annotations

import gc
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import statsmodels.api as sm

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"

# Default is "rolling": the per-filing reference windows every estimate in
# the paper uses. Set SURPRISE_SRC=topic to reproduce the retired
# per-calendar-year scoring, which the comparison appendix reports.
SRC = os.environ.get("SURPRISE_SRC", "rolling")
RES = REPO / "paper" / "results"

CLASSES = [f"{i:03d}" for i in range(1, 46)]
MIN_OBS = 500
GATE_LO, GATE_HI = 4.0, 8.5   # first maintenance window, registration-age years
GATE1_COHORTS = (2002, 2018)  # cohorts whose year-six window has elapsed
GATE2_COHORTS = (2002, 2013)  # cohorts whose year-ten window has elapsed


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def debut_panel() -> pl.DataFrame:
    """One row per owner: their first-ever filing, its scores, class and year."""
    parts = []
    for cls in CLASSES:
        tp = PROC / f"{SRC}_surprise_class{cls}_T200.parquet"
        tm = PROC / f"tm_class{cls}.parquet"
        if not (tp.exists() and tm.exists()):
            continue
        m = pl.read_parquet(
            tm, columns=["serial_number", "filing_date", "registration_date",
                         "owner_name", "goods_services", "status_code"],
        ).filter(
            pl.col("owner_name").is_not_null()
            & (pl.col("filing_date").fill_null("").str.len_chars() >= 8)
        ).with_columns(
            pl.col("filing_date").str.slice(0, 4).cast(pl.Int32, strict=False).alias("fyear"),
            pl.col("goods_services").str.len_chars().alias("gs_len"),
            pl.lit(cls).alias("nice_class"),
        ).drop("goods_services")

        t = pl.read_parquet(
            tp, columns=["serial_number", "topic_kl_vs_past",
                         "topic_kl_vs_future", "topic_dkl"],
        ).filter(pl.col("topic_dkl").is_finite())

        parts.append(m.join(t, on="serial_number", how="inner"))
        del m, t
        gc.collect()

    df = pl.concat(parts)
    del parts
    gc.collect()

    # The debut is the owner's earliest SCORED filing, not their earliest
    # filing: anything outside the interpretable range, or with too thin a
    # reference window, carries no score and cannot be the unit. Sorting on the
    # serial number after the year makes the pick deterministic when an owner
    # filed in several classes on the same day.
    df = df.sort(["owner_name", "fyear", "serial_number"]).unique(
        subset="owner_name", keep="first")
    log(f"[panel] {df.height:,} debut owners scored")
    return df


def gate_outcome(serials: pl.Series) -> pl.DataFrame:
    """Every dated maintenance cancellation for the given registered serials.

    Returns one row per event, unaggregated; main() computes registration age
    and applies the first-gate window (4.0-8.5 years). The SECOND gate is not a
    dated window at all -- a mark that dies at year six and one that dies at
    year ten share a terminal status, so gate two is identified among first-gate
    survivors by whether the registration was renewed (800) or died (710/900).
    Codes are Section 8 for domestic registrations and Section 71 for Madrid
    Section 66(a) extensions; reading only C8.. would score every 66(a)
    registration as a survivor whatever became of it.
    """
    c8 = pl.scan_parquet(PROC / "case_events.parquet").filter(
        (pl.col("code").is_in(["C8..", "C71T"])) & (pl.col("date") > 19000000)
    ).select("serial_number", "date").collect().with_columns(
        pl.col("date").cast(pl.Utf8).str.strptime(pl.Date, "%Y%m%d", strict=False)
        .alias("c8_d")
    ).drop_nulls("c8_d")
    return c8.filter(pl.col("serial_number").is_in(serials.implode()))


def zscore(a: np.ndarray) -> np.ndarray:
    return (a - np.nanmean(a)) / np.nanstd(a)


def fe_lpm(d: pd.DataFrame, y: str, xs: list[str], cell: str) -> dict:
    """LPM with cell fixed effects absorbed by within-transform, robust SE."""
    d = d[[y, cell] + xs].dropna()
    if len(d) < MIN_OBS:
        return {"skipped": "too few observations", "n": int(len(d))}
    g = d.groupby(cell)
    yd = d[y] - g[y].transform("mean")
    X = pd.DataFrame({x: d[x] - g[x].transform("mean") for x in xs}, index=d.index)
    try:
        res = sm.OLS(yd.astype(float), X.astype(float)).fit(cov_type="HC1")
    except Exception as exc:
        return {"skipped": f"{type(exc).__name__}: {exc}", "n": int(len(d))}
    out = {x: {"b_pp": float(res.params[x]) * 100,
               "se_pp": float(res.bse[x]) * 100,
               "t": float(res.tvalues[x])} for x in xs}
    out["n"] = int(len(d))
    out["n_cells"] = int(d[cell].nunique())
    out["base_rate_pct"] = float(d[y].mean()) * 100
    return out


def main() -> int:
    df = debut_panel()

    reg = df.filter(pl.col("registration_date").fill_null("").str.len_chars() >= 8)
    log(f"[panel] {reg.height:,} of them registered")

    c8 = gate_outcome(reg["serial_number"])
    reg = reg.with_columns(
        pl.col("registration_date").str.strptime(pl.Date, "%Y%m%d", strict=False)
        .alias("reg_d")
    )
    # Flag each maintenance window separately, then collapse to one row per
    # serial. A mark can appear once per cancellation event, so aggregate with
    # max() rather than joining the earliest date and testing it twice: the
    # earliest cancellation is the first-gate one, which would mask a
    # second-gate failure entirely.
    ages = reg.select("serial_number", "reg_d").join(
        c8, on="serial_number", how="inner").with_columns(
        ((pl.col("c8_d") - pl.col("reg_d")).dt.total_days() / 365.25).alias("age")
    ).with_columns(
        ((pl.col("age") >= GATE_LO) & (pl.col("age") < GATE_HI)).alias("f1"),
    ).group_by("serial_number").agg(pl.col("f1").max().alias("failed1"))

    reg = reg.join(ages, on="serial_number", how="left").with_columns(
        (~pl.col("failed1").fill_null(False)).cast(pl.Int8).alias("passed_gate"),
        pl.col("reg_d").dt.year().alias("reg_year"),
    ).with_columns(
        # Renewed (800) versus cancelled/expired (710/900). Only registrations
        # carrying one of those terminal statuses are informative; anything
        # else is still mid-life and is dropped by the null.
        pl.when(pl.col("status_code") == "800").then(1)
        .when(pl.col("status_code").is_in(["710", "900"])).then(0)
        .otherwise(None).cast(pl.Int8).alias("passed_gate2"),
    )

    fund_path = PROC / "funding_owner_match.parquet"
    funded_owners = None
    if fund_path.exists():
        funded_owners = pl.read_parquet(fund_path)
        log(f"[funding] {funded_owners.height:,} matched owner rows")
    else:
        log("[funding] funding_owner_match.parquet absent -- funding stages skipped")

    xw = pl.read_parquet(PROC / "uspto_sec_crosswalk.parquet",
                         columns=["owner_name"]).unique()

    # Listing markers. Two corrections matter here and both come from inspecting
    # the persisted Form D match rather than from theory:
    #   (1) 56% of owners carrying an exchange-registration marker registered
    #       BEFORE their first Reg D notice. Those are already-public companies
    #       doing private placements (PIPEs), not startups that raised and then
    #       listed. Conditioning a "listed given funded" outcome on the raw
    #       marker therefore measures the wrong thing, so the post-funding
    #       listing indicator requires ipo_date > first_formd_date.
    #   (2) The crosswalk marker keys on the exact owner-name string, so the
    #       same firm can be flagged under one spelling and not another
    #       (1,912 CIKs are internally inconsistent). The exchange-registration
    #       marker keys on CIK and has none, so it is preferred here.
    listed_after = set()
    if funded_owners is not None and {"in_8a", "ipo_date", "first_formd_date"} <= set(
            funded_owners.columns):
        la = funded_owners.with_columns(
            pl.col("ipo_date").cast(pl.Utf8), pl.col("first_formd_date").cast(pl.Utf8)
        ).filter(
            (pl.col("in_8a") == 1)
            & pl.col("ipo_date").is_not_null()
            & pl.col("first_formd_date").is_not_null()
            & (pl.col("ipo_date") > pl.col("first_formd_date"))
        )
        listed_after = set(la["owner_name"].to_list())
        log(f"[listing] {len(listed_after):,} owners listed AFTER their first round "
            f"(of {int(funded_owners['in_8a'].sum()):,} with any exchange marker)")

    base = df.select("owner_name", "nice_class", "fyear", "gs_len",
                     "topic_kl_vs_past", "topic_dkl",
                     "registration_date").to_pandas()
    base["cell"] = base.nice_class + "_" + base.fyear.astype(str)
    base["log_len"] = np.log1p(base.gs_len.fillna(0))
    # Atypicality is the AVERAGE of the two KL levels, not the past level alone.
    # dkl = past - future, so the average is past - dkl/2 and no extra column is
    # needed. (The past level alone was used through 2026-08-05; it correlates
    # with the average at 0.979, so this changes magnitudes only marginally.)
    base["atyp"] = base.topic_kl_vs_past.to_numpy() - 0.5 * base.topic_dkl.to_numpy()
    base["z_level"] = zscore(base.atyp.to_numpy())
    base["z_lean"] = zscore(base.topic_dkl.to_numpy())
    base["registered"] = base.registration_date.fillna("").str.len().ge(8).astype(int)
    # Listing is a property of the OWNER, matched on the exact USPTO
    # owner_name string in the SEC crosswalk. The crosswalk is
    # normalized-exact-match only, so this is a lower bound on true listing and
    # the base rate is correspondingly small; the estimand is a within-cell
    # contrast, which a uniform match rate leaves unbiased.
    base["listed"] = base.owner_name.isin(set(xw["owner_name"].to_list())).astype(int)
    base["listed_after_funding"] = base.owner_name.isin(listed_after).astype(int)
    if funded_owners is not None:
        base["funded"] = base.owner_name.isin(
            set(funded_owners["owner_name"].to_list())).astype(int)
    else:
        base["funded"] = np.nan

    gate = reg.select("owner_name", "passed_gate", "passed_gate2", "reg_year").to_pandas()
    base = base.merge(gate, on="owner_name", how="left")

    # Two unsigned operationalizations are reported at every stage, because the
    # paper uses both: atypicality A (the average of the two KL levels)
    # and |dKL| (the magnitude of the signed deviation, used in its published
    # decomposition). They are related but not identical, and reporting both
    # keeps this table comparable to Table~\ref{tab:edgar}.
    base["abs_lean"] = np.abs(base["z_lean"])
    SPEC_LEVEL = ["z_level", "z_lean", "log_len"]
    SPEC_ABS = ["abs_lean", "z_lean", "log_len"]
    stages: list[dict] = []

    def add(label, cond_label, frame, outcome, years=None):
        f = frame if years is None else frame[frame.fyear.between(*years)]
        r = fe_lpm(f, outcome, SPEC_LEVEL, "cell")
        r_abs = fe_lpm(f, outcome, SPEC_ABS, "cell")
        r["spec_absdkl"] = r_abs
        r.update({"stage": label, "conditioned_on": cond_label,
                  "cohort": f"{years[0]}--{years[1]}" if years else "all scored"})
        stages.append(r)
        if "skipped" not in r:
            extra = ""
            if "skipped" not in r_abs:
                extra = (f"  |dKL|={r_abs['abs_lean']['b_pp']:+.4f}"
                         f"(t={r_abs['abs_lean']['t']:+.1f})")
            log(f"[stage] {label:24s} | {cond_label:12s} n={r['n']:>9,} "
                f"base={r['base_rate_pct']:5.2f}%  level={r['z_level']['b_pp']:+.4f}"
                f"(t={r['z_level']['t']:+.1f})  lean={r['z_lean']['b_pp']:+.4f}"
                f"(t={r['z_lean']['t']:+.1f}){extra}")

    add("Reached registration", "filed", base, "registered")
    # Restricted to elapsed cohorts. Without this, registrations whose
    # 4.0-8.5 year window has not opened carry no cancellation event and are
    # coded as survivors -- 35% of registrations, and 2.0M whose window has
    # not started at all, which passes them mechanically.
    g1 = base[(base.registered == 1) & base.reg_year.between(*GATE1_COHORTS)]
    add("Passed first gate (yr 6)", "registered", g1, "passed_gate")
    g2 = base[(base.registered == 1) & (base.passed_gate == 1)
              & base.reg_year.between(*GATE2_COHORTS)]
    add("Passed second gate (yr 10)", "passed first", g2, "passed_gate2")
    if funded_owners is not None:
        add("Raised a Reg D round", "filed", base, "funded", years=(2009, 2018))
        add("Raised a Reg D round", "registered",
            base[base.registered == 1], "funded", years=(2009, 2018))
        add("Listed after the round", "funded",
            base[base.funded == 1], "listed_after_funding", years=(2009, 2018))
    add("Reached listed universe", "registered",
        base[base.registered == 1], "listed")

    out = {"scoring": SRC,
           "gate1_cohorts": list(GATE1_COHORTS), "gate2_cohorts": list(GATE2_COHORTS),
           "spec": ("outcome ~ z_level + z_lean + log_len; class x debut-year FE; "
                    "HC1 robust SE; coefficients in percentage points"),
           "z_level": "standardized (KL vs past + KL vs future)/2 (atypicality)",
           "z_lean": "standardized dKL = KL(vs past) - KL(vs future) (lead)",
           "n_debut_owners": int(df.height),
           "stages": stages}
    (RES / "staged_outcomes.json").write_text(json.dumps(out, indent=1))

    rows = [s for s in stages if "skipped" not in s]
    L = [r"\begin{table}[h]\centering\small",
         r"\caption{Every outcome in the funnel, conditioned on its prerequisite, "
         r"under one specification. Unit is the debut owner, scored on that owner's "
         r"first filing. Each row is a linear probability model with class$\times$"
         r"debut-year fixed effects and heteroskedasticity-robust errors; "
         r"coefficients are percentage points per standard deviation. "
         r"\emph{Atypicality} is the average of the two KL levels; \emph{lead} "
         r"is the signed difference. Cohort windows differ because each maintenance "
         r"gate needs elapsed time and Form~D is electronic only from 2009, so the "
         r"rows are not a nested decomposition. The first gate is a dated "
         r"cancellation for non-use; the second is whether a registration that "
         r"cleared the first was renewed rather than cancelled or expired, "
         r"restricted to 2002--2013 cohorts, since a year-six and a year-ten death "
         r"share a terminal status and are separable only this way. This table "
         r"alone is estimated at $T = 200$ topics rather than the $T = 50$ used "
         r"elsewhere; the sign and ordering of every row are unchanged at $T = 50$.}",
         r"\label{tab:staged}",
         r"\begin{tabular}{llrrrrr}", r"\toprule",
         r" & & & & \multicolumn{2}{c}{Unsigned} & Signed \\",
         r"\cmidrule(lr){5-6}\cmidrule(lr){7-7}",
         r"Outcome & Given & $n$ & Base & Atypicality & $|\Delta$KL$|$ & Lead \\",
         r"\midrule"]
    for r in rows:
        a = r.get("spec_absdkl", {})
        acell = (f"${a['abs_lean']['b_pp']:+.3f}$ ({a['abs_lean']['se_pp']:.3f})"
                 if "abs_lean" in a else "---")
        L.append(f"{r['stage']} & {r['conditioned_on']} & {r['n']:,} & "
                 f"{r['base_rate_pct']:.2f}\\% & "
                 f"${r['z_level']['b_pp']:+.3f}$ ({r['z_level']['se_pp']:.3f}) & "
                 f"{acell} & "
                 f"${r['z_lean']['b_pp']:+.3f}$ ({r['z_lean']['se_pp']:.3f}) \\\\")
    L += [r"\bottomrule", r"\end{tabular}",
          r"\\[0.3em]\footnotesize Standard errors in parentheses, in place of "
          r"significance stars; see the note to Table~\ref{tab:gate_lpm}.",
          r"\end{table}"]
    (RES / "staged_outcomes_table.tex").write_text("\n".join(L), encoding="utf-8")
    log(f"[done] -> {RES/'staged_outcomes.json'} and staged_outcomes_table.tex")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
