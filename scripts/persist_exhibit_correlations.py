"""Persist the K-/K+ correlations the construct section quotes.

Computes, on the production rolling scoring:
  1. corr(K-, K+) on the complete scored class-009 corpus, with n.
  2. corr(A, L) and corr(K- , L) on the same frame (the rotation's payoff).
  3. sd of both levels and of L.

Output: paper/results/exhibit_correlations.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
RES = REPO / "paper" / "results"


def main() -> int:
    d = pl.read_parquet(PROC / "rolling_surprise_class009.parquet",
                        columns=["topic_kl_vs_past", "topic_kl_vs_future", "topic_dkl"]).filter(
        pl.col("topic_dkl").is_finite())
    kp = d["topic_kl_vs_past"].to_numpy()
    kf = d["topic_kl_vs_future"].to_numpy()
    L = kp - kf
    A = (kp + kf) / 2
    out = {
        "class": "009",
        "n": int(d.height),
        "corr_Kpast_Kfuture": float(np.corrcoef(kp, kf)[0, 1]),
        "corr_A_L": float(np.corrcoef(A, L)[0, 1]),
        "corr_Kpast_L": float(np.corrcoef(kp, L)[0, 1]),
        "sd_Kpast": float(kp.std()),
        "sd_Kfuture": float(kf.std()),
        "sd_L": float(L.std()),
        "mean_Kpast": float(kp.mean()),
        "mean_Kfuture": float(kf.mean()),
    }
    (RES / "exhibit_correlations.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
