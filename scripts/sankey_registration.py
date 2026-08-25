"""A flow diagram of the registration step, split by who drafted the filing.

Left nodes: counsel-represented and self-filed applications (attorney of
record read from the prosecution history). Right nodes: reached
registration, abandoned. Unique serials, filing years 1995-2018. Ribbons
drawn with bezier patches; widths proportional to counts.

Outputs: paper/results/fig_registration_sankey.png
         paper/results/fig_registration_sankey.json
"""
from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath
import polars as pl

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
RES = REPO / "paper" / "results"
CLASSES = [f"{i:03d}" for i in range(1, 46)]


def log(m): print(m, file=sys.stderr, flush=True)


def ribbon(ax, y0, h0, y1, h1, color, alpha=0.55):
    x0, x1 = 0.22, 0.78
    verts = [(x0, y0), ((x0 + x1) / 2, y0), ((x0 + x1) / 2, y1), (x1, y1),
             (x1, y1 + h1), ((x0 + x1) / 2, y1 + h1), ((x0 + x1) / 2, y0 + h0), (x0, y0 + h0),
             (x0, y0)]
    codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
             MplPath.LINETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
             MplPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MplPath(verts, codes), facecolor=color, edgecolor="none", alpha=alpha))


def main() -> int:
    att = pl.read_parquet(PROC / "case_extras.parquet", columns=["serial_number", "attorney_name"]).with_columns(
        (pl.col("attorney_name").fill_null("").str.len_chars() > 0).alias("rep"))
    parts = []
    for c in CLASSES:
        p = PROC / f"tm_class{c}.parquet"
        if not p.exists():
            continue
        parts.append(pl.read_parquet(p, columns=["serial_number", "filing_date", "registration_date"]))
    d = pl.concat(parts).unique("serial_number"); del parts; gc.collect()
    d = d.filter(pl.col("filing_date").fill_null("").str.len_chars() >= 8).with_columns(
        pl.col("filing_date").str.slice(0, 4).cast(pl.Int32, strict=False).alias("fy"),
        (pl.col("registration_date").fill_null("").str.len_chars() >= 8).alias("reg"))
    d = d.filter(pl.col("fy").is_between(1995, 2018)).join(att, on="serial_number", how="left").with_columns(
        pl.col("rep").fill_null(False))
    n = d.height
    cnt = {(r, g): d.filter((pl.col("rep") == r) & (pl.col("reg") == g)).height
           for r in (True, False) for g in (True, False)}
    out = {"n": n, "counts": {f"{'rep' if r else 'self'}_{'reg' if g else 'ab'}": v
                              for (r, g), v in cnt.items()}}
    log(json.dumps(out))

    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    GAP = 0.04
    rep_n = cnt[(True, True)] + cnt[(True, False)]
    self_n = cnt[(False, True)] + cnt[(False, False)]
    reg_n = cnt[(True, True)] + cnt[(False, True)]
    ab_n = n - reg_n
    scale = (1.0 - GAP) / n
    # left nodes: represented on top
    yl_rep = GAP + self_n * scale
    yl_self = 0.0
    # right nodes: registered on top
    yr_reg = GAP + ab_n * scale
    yr_ab = 0.0
    # ribbons: order within nodes -- registered part on top of each left node
    ribbon(ax, yl_rep + cnt[(True, False)] * scale, cnt[(True, True)] * scale,
           yr_reg + cnt[(False, True)] * scale, cnt[(True, True)] * scale, "#2b6cb0")
    ribbon(ax, yl_rep, cnt[(True, False)] * scale,
           yr_ab + cnt[(False, False)] * scale, cnt[(True, False)] * scale, "#9db8d2")
    ribbon(ax, yl_self + cnt[(False, False)] * scale, cnt[(False, True)] * scale,
           yr_reg, cnt[(False, True)] * scale, "#c0392b")
    ribbon(ax, yl_self, cnt[(False, False)] * scale,
           yr_ab, cnt[(False, False)] * scale, "#d9a09a")
    for x, y, h, lab in ((0.20, yl_rep, rep_n * scale, f"counsel-represented\n{rep_n:,} ({100*rep_n/n:.1f}%)"),
                         (0.20, yl_self, self_n * scale, f"self-filed\n{self_n:,} ({100*self_n/n:.1f}%)"),
                         (0.80, yr_reg, reg_n * scale, f"reached registration\n{reg_n:,} ({100*reg_n/n:.1f}%)"),
                         (0.80, yr_ab, ab_n * scale, f"abandoned\n{ab_n:,} ({100*ab_n/n:.1f}%)")):
        ax.add_patch(plt.Rectangle((x if x < 0.5 else x, y), 0.02, h, color="#333333"))
        ax.text(x - 0.012 if x < 0.5 else x + 0.032, y + h / 2, lab, fontsize=9,
                ha="right" if x < 0.5 else "left", va="center")
    for r, g, share, xfrac in ((True, True, cnt[(True, True)] / rep_n, 0.34),
                               (False, True, cnt[(False, True)] / self_n, 0.34)):
        pass
    ax.text(0.5, yl_rep + cnt[(True, False)] * scale + cnt[(True, True)] * scale / 2,
            f"{100*cnt[(True, True)]/rep_n:.1f}% of represented", fontsize=8, ha="center", va="center")
    ax.text(0.5, yl_self + cnt[(False, False)] * scale + cnt[(False, True)] * scale / 2,
            f"{100*cnt[(False, True)]/self_n:.1f}% of self-filed", fontsize=8, ha="center", va="center")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(-0.02, 1.02 + GAP)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(RES / "fig_registration_sankey.png", dpi=150, bbox_inches="tight")
    (RES / "fig_registration_sankey.json").write_text(json.dumps(out, indent=1))
    log("[done]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
