"""Convergent / discriminant validity battery for dKL.

(1) Reframe the patent correlation honestly: r_raw=0.010, r_log=-0.009 (already
    computed) + Spearman rank correlation as a robustness (rank-based, immune to
    the heavy patent tail).
(2) Discriminant validity against THREE external, non-patent, non-financial
    innovation lists (BCG Most Innovative 2021-23, MIT TR50 2015, Crunchbase):
    do firms an independent panel calls "most innovative" score higher on dKL
    than firms that panel did NOT list? If dKL is an innovation measure, listed
    firms should sit above the non-listed baseline.

Inputs : data_publish/firm_year_patents_and_dkl.csv
         data_publish/firm_year_dkl.csv
         data_publish/comparators/external_crosswalk.csv
         paper/results/patent_compare_metrics.json
Output : paper/results/wsD_validity_battery.json (+ stdout)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
PUB = REPO / "data_publish"
RES = REPO / "paper" / "results"
OUT = RES / "wsD_validity_battery.json"


def main() -> int:
    results = {}

    # ---- (1) patent correlation: reframe + rank robustness -----------------
    pc = json.loads((RES / "patent_compare_metrics.json").read_text())
    pat = pd.read_csv(PUB / "firm_year_patents_and_dkl.csv")
    # cross-sectional firm-year Spearman (rank) corr, robust to the patent tail
    sp = stats.spearmanr(pat.mean_dkl, pat.n_patents)
    results["patent_corr"] = {
        "r_pearson_raw": pc["r_fy_raw"], "r_pearson_log": pc["r_fy_log"],
        "r_spearman": float(sp.statistic), "spearman_p": float(sp.pvalue),
        "n": int(len(pat)),
    }
    print("=== (1) patent correlation (cross-sectional) ===")
    print(f"  Pearson raw   = {pc['r_fy_raw']:+.4f}")
    print(f"  Pearson log   = {pc['r_fy_log']:+.4f}")
    print(f"  Spearman rank = {sp.statistic:+.4f}  (p={sp.pvalue:.3g}, n={len(pat):,})")

    # ---- population baseline: firm-level mean dKL (public, CIK-linked) ------
    fy = pd.read_csv(PUB / "firm_year_dkl.csv")
    fy["cik"] = fy.cik.astype(str)
    w = fy.n_filings.clip(lower=1)
    fy["_wd"] = fy.mean_dkl * w
    firm = fy.groupby("cik").apply(
        lambda g: pd.Series({"mean_dkl": g._wd.sum() / g.n_filings.clip(lower=1).sum(),
                             "sec_name": g.sec_name.iloc[0]}),
        include_groups=False,
    ).reset_index()
    pop_mean, pop_sd = firm.mean_dkl.mean(), firm.mean_dkl.std()
    results["population"] = {"n_firms": int(len(firm)), "mean_dkl": float(pop_mean),
                             "sd_dkl": float(pop_sd)}
    print(f"\n=== population (CIK-linked public firms) ===")
    print(f"  n_firms={len(firm):,}  mean dKL={pop_mean:+.4f}  sd={pop_sd:.4f}")

    # ---- (2) discriminant validity vs external lists -----------------------
    xw = pd.read_csv(PUB / "comparators" / "external_crosswalk.csv")
    xw["matched_cik"] = xw.matched_cik.apply(
        lambda v: str(int(float(v))) if str(v).strip() not in ("", "nan") else "")
    xw["matched_firm_mean_dkl"] = pd.to_numeric(xw.matched_firm_mean_dkl, errors="coerce")
    matched = xw[xw.matched_firm_mean_dkl.notna()].copy()
    print(f"\n=== (2) external lists: {len(xw)} list-rows, {len(matched)} matched to a firm dKL ===")

    listed_ciks_all = set(matched.matched_cik)
    results["lists"] = {}
    for comp, g in matched.groupby("comparator"):
        listed = g.matched_firm_mean_dkl.values
        listed_ciks = set(g.matched_cik)
        base = firm[~firm.cik.isin(listed_ciks)].mean_dkl.values  # non-listed baseline
        mw = stats.mannwhitneyu(listed, base, alternative="two-sided")
        d = (listed.mean() - base.mean()) / pop_sd  # effect size in pop-sd units
        # percentile of listed median within population
        pct = float((firm.mean_dkl < np.median(listed)).mean() * 100)
        results["lists"][comp] = {
            "n_matched": int(len(listed)),
            "listed_mean_dkl": float(listed.mean()),
            "listed_median_dkl": float(np.median(listed)),
            "baseline_mean_dkl": float(base.mean()),
            "diff_in_pop_sd": float(d),
            "mannwhitney_p": float(mw.pvalue),
            "listed_median_pctile": pct,
        }
        print(f"\n  -- {comp}")
        print(f"     n={len(listed)}  listed mean dKL={listed.mean():+.4f} "
              f"(median {np.median(listed):+.4f}, ~{pct:.0f}th pctile of pop)")
        print(f"     baseline mean={base.mean():+.4f}  diff={d:+.3f} pop-sd  "
              f"MannWhitney p={mw.pvalue:.3g}")

    # pooled across all lists (union of matched firms)
    listed = matched.groupby("matched_cik").matched_firm_mean_dkl.mean()
    base = firm[~firm.cik.isin(listed_ciks_all)].mean_dkl.values
    mw = stats.mannwhitneyu(listed.values, base, alternative="two-sided")
    d = (listed.mean() - base.mean()) / pop_sd
    results["lists"]["POOLED"] = {
        "n_matched": int(len(listed)), "listed_mean_dkl": float(listed.mean()),
        "baseline_mean_dkl": float(base.mean()), "diff_in_pop_sd": float(d),
        "mannwhitney_p": float(mw.pvalue),
    }
    print(f"\n  -- POOLED (union of all lists)")
    print(f"     n={len(listed)}  listed mean={listed.mean():+.4f}  baseline={base.mean():+.4f}")
    print(f"     diff={d:+.3f} pop-sd  MannWhitney p={mw.pvalue:.3g}")

    OUT.write_text(json.dumps(results, indent=2))
    print(f"\n[done] -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
