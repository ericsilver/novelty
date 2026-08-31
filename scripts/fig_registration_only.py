"""Registration figures that show registration and nothing else.

Panel figure 1 (fig_registration_deciles.png):
  A: share reaching registration by within-class-and-year decile of
     atypicality, all filings 1995-2018, pooled, with three named
     industries, all with binomial 95% bands.
  B: the same by decile of lead.

Figure 2 (fig_registration_plane.png): the K-/K+ plane for classes 009 and
035, hexbinned, coloured by registration rate, with iso-rate contours,
corner labels (leading/lagging, typical/atypical), and named filings
marked. Registration only; no later outcome.

Outputs: paper/results/fig_registration_deciles.png
         paper/results/fig_registration_plane.png
         paper/results/fig_registration_only.json
"""
from __future__ import annotations

import gc
import json
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
CLASSES = [f"{i:03d}" for i in range(1, 46)]
SHOW = {"009": "Software & Electronics", "025": "Clothing & Footwear", "035": "Advertising & Retail"}
# serial, label, label offset (points). Serials resolved against owner of
# record; the IPOD and AMAZON.COM serials are the ones asserted in
# representation_appendix.py.
EXEMPLARS = {
    "009": [("75982871", "IPOD (2001)", (10, -14)),
            ("86359718", "ETHEREUM (2014)", (10, 2)),
            ("77318565", "ANDROID (2007)", (-86, 6)),
            ("75493408", "TIVO (1998)", (8, -16))],
    "035": [("75277670", "AMAZON.COM (1997)", (10, -14)),
            ("76314811", "GOOGLE (2001)", (-72, 8)),
            ("85023193", "AIRBNB (2010)", (14, -4)),
            ("75412591", "NETFLIX (1997)", (10, 2))],
}
NDEC = 10


def log(m): print(m, file=sys.stderr, flush=True)


def main() -> int:
    parts = []
    for c in CLASSES:
        tp, sp = PROC / f"tm_class{c}.parquet", PROC / f"rolling_surprise_class{c}.parquet"
        if not (tp.exists() and sp.exists()):
            continue
        tm = pl.read_parquet(tp, columns=["serial_number", "filing_date", "registration_date"]).filter(
            pl.col("filing_date").fill_null("").str.len_chars() >= 8).with_columns(
            pl.col("filing_date").str.slice(0, 4).cast(pl.Int32, strict=False).alias("fy"),
            (pl.col("registration_date").fill_null("").str.len_chars() >= 8).cast(pl.Float64).alias("reg"))
        sc = pl.read_parquet(sp, columns=["serial_number", "topic_kl_vs_past", "topic_kl_vs_future"]).filter(
            pl.col("topic_kl_vs_past").is_finite() & pl.col("topic_kl_vs_future").is_finite())
        parts.append(tm.join(sc, on="serial_number", how="inner").with_columns(pl.lit(c).alias("cls")))
        del tm, sc
        gc.collect()
    d = pl.concat(parts).filter(pl.col("fy").is_between(1995, 2018)); del parts; gc.collect()
    d = d.with_columns(((pl.col("topic_kl_vs_past") + pl.col("topic_kl_vs_future")) / 2).alias("A"),
                       (pl.col("topic_kl_vs_past") - pl.col("topic_kl_vs_future")).alias("L"))
    log(f"[frame] {d.height:,} scored class-records 1995-2018")

    def decile_curve(df, var):
        s = df.sort(["cls", "fy", var, "serial_number"]).with_columns(
            ((pl.col(var).rank("ordinal").over(["cls", "fy"]) - 1) * NDEC // pl.len().over(["cls", "fy"]))
            .cast(pl.Int8).alias("q"))
        g = s.group_by("q").agg(pl.col("reg").mean().alias("p"), pl.len().alias("n")).sort("q")
        return ([float(v) for v in g["p"]], [float((p * (1 - p) / n) ** 0.5) for p, n in zip(g["p"], g["n"])])

    out = {"n": int(d.height), "pooled": {}, "industries": {}}
    for var in ("A", "L"):
        out["pooled"][var] = decile_curve(d, var)
    for c in SHOW:
        out["industries"][c] = {v: decile_curve(d.filter(pl.col("cls") == c), v) for v in ("A", "L")}

    xs = [i + 1 for i in range(NDEC)]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True)
    colors = {"009": "#2b6cb0", "025": "#c0392b", "035": "#2f855a"}
    for ax, var, lab in ((axes[0], "A", "atypicality"), (axes[1], "L", "lead")):
        p, se = out["pooled"][var]
        ax.fill_between(xs, [100 * (a - 1.96 * b) for a, b in zip(p, se)],
                        [100 * (a + 1.96 * b) for a, b in zip(p, se)], color="#444", alpha=0.15, lw=0)
        ax.plot(xs, [100 * v for v in p], "o-", color="#222", lw=2.2, ms=4, label="all classes")
        for c, name in SHOW.items():
            pc, sec_ = out["industries"][c][var]
            ax.fill_between(xs, [100 * (a - 1.96 * b) for a, b in zip(pc, sec_)],
                            [100 * (a + 1.96 * b) for a, b in zip(pc, sec_)],
                            color=colors[c], alpha=0.13, lw=0)
            ax.plot(xs, [100 * v for v in pc], "-", color=colors[c], lw=1.4, alpha=0.9, label=name)
        ax.set_xlabel(f"decile of {lab}, within Nice class and filing year")
        ax.set_xticks(xs)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("% of class-records reaching registration")
    axes[0].legend(fontsize=8, frameon=False, loc="lower center")
    fig.tight_layout()
    fig.savefig(RES / "fig_registration_deciles.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log("[fig] deciles done")

    # Plane figure: K-/K+ hexbin coloured by registration rate, with contours.
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.4))
    for ax, c in zip(axes, ("009", "035")):
        sub = d.filter(pl.col("cls") == c)
        x = sub["topic_kl_vs_past"].to_numpy(); y = sub["topic_kl_vs_future"].to_numpy()
        z = sub["reg"].to_numpy()
        lo, hi = np.percentile(np.r_[x, y], [0.5, 99.5])
        hb = ax.hexbin(x, y, C=z, reduce_C_function=np.mean, gridsize=42, mincnt=50,
                       extent=(lo, hi, lo, hi), cmap="RdYlGn", vmin=0.35, vmax=0.75)
        # iso-rate contours on a smoothed grid
        NB = 36
        xe = np.linspace(lo, hi, NB + 1)
        xi = np.clip(np.digitize(x, xe) - 1, 0, NB - 1)
        yi = np.clip(np.digitize(y, xe) - 1, 0, NB - 1)
        s_ = np.zeros((NB, NB)); n_ = np.zeros((NB, NB))
        np.add.at(s_, (yi, xi), z); np.add.at(n_, (yi, xi), 1.0)
        with np.errstate(invalid="ignore"):
            m = s_ / n_
        # 3x3 count-weighted smoothing, twice
        for _ in range(2):
            sm = np.zeros_like(s_); nm = np.zeros_like(n_)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    w = 1.0 if (dx or dy) else 2.0
                    sm += w * np.roll(np.roll(np.nan_to_num(m * n_), dy, 0), dx, 1)
                    nm += w * np.roll(np.roll(n_, dy, 0), dx, 1)
            with np.errstate(invalid="ignore"):
                m = np.where(nm > 0, sm / nm, np.nan)
        cx = (xe[:-1] + xe[1:]) / 2
        mask = n_ < 30
        mm = np.ma.masked_array(m, mask=mask | ~np.isfinite(m))
        cs = ax.contour(cx, cx, mm, levels=[0.45, 0.55, 0.65], colors="#222",
                        linewidths=0.9, alpha=0.8)
        ax.clabel(cs, fmt=lambda v: f"{100*v:.0f}%", fontsize=7)
        ax.plot([lo, hi], [lo, hi], ls="--", color="#333", lw=1)
        for serial, label, (ox, oy) in EXEMPLARS[c]:
            row = sub.filter(pl.col("serial_number") == serial)
            if not row.height:
                log(f"[warn] exemplar {serial} {label} not scored in {c}")
                continue
            r = row.row(0, named=True)
            ax.scatter([r["topic_kl_vs_past"]], [r["topic_kl_vs_future"]], s=70,
                       facecolor="none", edgecolor="#111", lw=1.6, zorder=5)
            ax.annotate(label, (r["topic_kl_vs_past"], r["topic_kl_vs_future"]),
                        xytext=(ox, oy), textcoords="offset points", fontsize=8,
                        arrowprops=dict(arrowstyle="-", lw=0.8, color="#111"))
        ax.text(0.98, 0.02, "leading\n($K^-$ high, $K^+$ low)", fontsize=8,
                color="#333", ha="right", va="bottom", transform=ax.transAxes)
        ax.text(0.02, 0.98, "lagging\n($K^+$ high, $K^-$ low)", fontsize=8,
                color="#333", ha="left", va="top", transform=ax.transAxes)
        ax.text(0.02, 0.02, "typical", fontsize=8, color="#333",
                ha="left", va="bottom", transform=ax.transAxes)
        ax.text(0.98, 0.98, "atypical", fontsize=8, color="#333",
                ha="right", va="top", transform=ax.transAxes)
        ax.set_xlabel("past-facing surprise $K^-$ (nats)")
        ax.set_ylabel("future-facing surprise $K^+$ (nats)")
        ax.set_title({"009": "Software & Electronics (009)", "035": "Advertising & Retail (035)"}[c], fontsize=10)
    cb = fig.colorbar(hb, ax=axes, fraction=0.03, pad=0.02)
    cb.set_label("share reaching registration")
    fig.savefig(RES / "fig_registration_plane.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    (RES / "fig_registration_only.json").write_text(json.dumps(out, indent=1))
    log("[done] registration figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
