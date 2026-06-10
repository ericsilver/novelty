"""Returns-instability diagnostic for the headline "+28pp per sigma" claim.

We cannot re-estimate value-weighted / delisting-adjusted here (the returns
panel data/processed/stock_returns.parquet is not on disk). But the existing
returns_metrics.json is enough to show WHY the debut 4y excess-return effect is
not credible as a headline: its horizon profile is non-monotone (peaks at 4y,
collapses at 5y) and it is ~7x the firm-year effect estimated on 11x more data.
Both are signatures of a small-sample right-tail artifact, not a return premium.

The coefficient is per-sigma (returns_regression.py z-scores the predictor),
so +0.284 really is +28pp per sigma of debut dKL.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "paper" / "results"
OUT = RES / "wsE_returns_diagnostic.json"


def main() -> int:
    m = json.loads((RES / "returns_metrics.json").read_text())
    rows = m["rows"]

    def get(spec, outcome):
        for r in rows:
            if r["spec"] == spec and r["predictor"] == "mean_dkl" and r["outcome"] == outcome:
                return r
        return None

    print(f"panels: firm-year n={m['panel_a_n']:,}  debut n={m['panel_b_n']:,}\n")
    print(f"{'horizon':>7} | {'debut excess':>22} | {'firm-year excess':>22} | ratio")
    print("-" * 70)
    profile = {}
    for k in (1, 2, 3, 4, 5):
        db = get("debut", f"excess_{k}y")
        fy = get("firm_year", f"excess_{k}y")
        if not (db and fy):
            continue
        db_t = db["coef"] / db["se"]
        fy_t = fy["coef"] / fy["se"]
        ratio = db["coef"] / fy["coef"] if fy["coef"] else float("nan")
        profile[k] = {
            "debut_coef": db["coef"], "debut_se": db["se"], "debut_t": db_t, "debut_n": db["n"],
            "fy_coef": fy["coef"], "fy_se": fy["se"], "fy_t": fy_t, "fy_n": fy["n"],
            "ratio_debut_over_fy": ratio,
        }
        print(f"{k:>5}y  | {db['coef']:+.3f} (SE {db['se']:.3f}, t {db_t:+.1f}) "
              f"| {fy['coef']:+.3f} (SE {fy['se']:.3f}, t {fy_t:+.1f}) | {ratio:5.1f}x")

    # non-monotonicity flag: debut profile should be ~monotone for a real premium
    dc = [profile[k]["debut_coef"] for k in sorted(profile)]
    peak_k = sorted(profile)[dc.index(max(dc))]
    non_monotone = not (dc == sorted(dc) or dc == sorted(dc, reverse=True))
    verdict = {
        "coef_is_per_sigma": True,
        "debut_excess_profile": dc,
        "peak_horizon": peak_k,
        "non_monotone": non_monotone,
        "debut_4y_over_firmyear_4y": profile[4]["ratio_debut_over_fy"],
        "recommendation": (
            "Drop the +28pp/sigma debut figure from the headline. It peaks at 4y "
            "and collapses at 5y, and is ~7x the firm-year estimate on 11x less data "
            "-- a small-sample right-tail signature. Report the firm-year effect "
            "(+4pp/sigma at 4y, t~2.5, n=16k) as the defensible financial association, "
            "pending value-weighted, delisting-adjusted, 5/95-winsorized re-estimation."
        ),
    }
    print(f"\npeak at {peak_k}y; non-monotone={non_monotone}; "
          f"debut/firm-year @4y = {profile[4]['ratio_debut_over_fy']:.1f}x")
    OUT.write_text(json.dumps({"profile": profile, "verdict": verdict}, indent=2))
    print(f"[done] -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
