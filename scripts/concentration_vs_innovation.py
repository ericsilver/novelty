"""Does ethnic/surname concentration correlate with innovation?

Two parallel tests:
  (A) TRADEMARK SIDE -- per NICE class:
      - x: ethnic concentration = max(ethnic share) across the four
           large ethnic buckets (White, Black, API, Hispanic), or
           1 - Shannon entropy normalized over those four
      - y: mean ΔKL across individual-name debut filings in the class
      - Pearson + Spearman correlation
  (B) PATENT SIDE -- per CPC class:
      - x: surname HHI in the 2010s (concentration)
      - y: patent volume growth 2010s → 2020s (proxy for innovation
           activity; replicates Hsieh-Hurst-Jones-Klenow 2019 in
           narrower lens)

Reads: paper/results/ethnic_cluster_dkl_panel.csv,
       paper/results/patent_guild_hhi_panel.csv

Outputs:
  paper/results/concentration_vs_innovation.png
  paper/results/concentration_vs_innovation_tm.csv      (per-class TM panel)
  paper/results/concentration_vs_innovation_pv.csv      (per-class PV panel)
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "paper" / "results"

INDUSTRY_NAMES = {
    "001":"Chemicals","002":"Paints","003":"Cosmetics & Cleaning",
    "004":"Oils & Fuels","005":"Pharmaceuticals","006":"Common Metals",
    "007":"Machines","008":"Hand Tools","009":"Electronics & Software",
    "010":"Surgical/Medical/Dental","011":"Lighting/Heating","012":"Vehicles",
    "013":"Firearms","014":"Jewelry","015":"Musical Instruments",
    "016":"Paper/Print","017":"Rubber/Plastic","018":"Leather",
    "019":"Building Materials","020":"Furniture","021":"Household Utensils",
    "022":"Ropes/Nets/Textiles","023":"Yarns","024":"Fabrics",
    "025":"Clothing & Footwear","026":"Lace/Embroidery","027":"Carpets",
    "028":"Toys & Sports","029":"Meat/Fish/Dairy","030":"Coffee/Tea/Sugar",
    "031":"Agricultural Products","032":"Beverages (Non-Alcoholic)",
    "033":"Alcoholic Beverages","034":"Tobacco",
    "035":"Advertising & Retail","036":"Insurance & Financial",
    "037":"Construction & Repair","038":"Telecommunications",
    "039":"Transport & Travel","040":"Materials Treatment",
    "041":"Education & Entertainment","042":"Scientific & Tech Services",
    "043":"Hotel/Restaurant/Food","044":"Medical & Veterinary",
    "045":"Legal & Personal",
}


def tm_concentration_innovation():
    """Per NICE class: ethnic concentration (max share over 4 big buckets)
    vs mean ΔKL across all individual-name debut filers in the class."""
    panel = pd.read_csv(OUT / "ethnic_cluster_dkl_panel.csv")
    # Filter to the four big ethnic groups
    big4 = ["White","Black","API","Hispanic"]
    sub = panel[panel["ethnic"].isin(big4)].copy()
    # For each class: ethnic shares + n
    cells = sub.pivot_table(index="class_id", columns="ethnic",
                             values=["share_in_class","n_eff","mean_dkl"],
                             aggfunc="first")
    # Class-level total ethnic ΔKL: weighted by n_eff
    out_rows = []
    for cls, row in cells.iterrows():
        shares = np.array([row.get(("share_in_class", e), 0) for e in big4])
        ns     = np.array([row.get(("n_eff", e), 0) for e in big4])
        dkls   = np.array([row.get(("mean_dkl", e), np.nan) for e in big4])
        if ns.sum() < 500:
            continue
        # Renormalize shares to sum to 1 across the 4 (Mixed/Two+/AIAN absorbed elsewhere)
        s_norm = shares / shares.sum() if shares.sum() > 0 else shares
        max_share = s_norm.max()
        # Entropy of the 4-vector
        s_safe = s_norm[s_norm > 0]
        entropy = -(s_safe * np.log(s_safe)).sum()
        norm_entropy = entropy / np.log(4)
        concentration = 1 - norm_entropy   # 0 = uniform, 1 = single group
        # Class-level weighted-mean ΔKL across the 4 groups
        valid = ~np.isnan(dkls)
        mean_dkl = (dkls[valid] * ns[valid]).sum() / ns[valid].sum() if valid.any() else np.nan
        out_rows.append({
            "class_id": cls,
            "name": INDUSTRY_NAMES.get(cls, cls),
            "n_eff_total": ns.sum(),
            "max_share": max_share,
            "concentration": concentration,
            "mean_dkl": mean_dkl,
            "dominant_eth": big4[int(s_norm.argmax())] if shares.sum() > 0 else None,
        })
    df = pd.DataFrame(out_rows).dropna().sort_values("concentration", ascending=False)
    df.to_csv(OUT / "concentration_vs_innovation_tm.csv", index=False)
    return df


def pv_concentration_innovation():
    """Per CPC class: surname HHI in the 2010s vs patent volume
    growth 2010s → 2020s."""
    panel = pd.read_csv(OUT / "patent_guild_hhi_panel.csv")
    piv = panel.pivot_table(index="cpc_class", columns="decade",
                             values=["hhi","total_inventors"])
    rows = []
    for cls, r in piv.iterrows():
        hhi_2010 = r.get(("hhi", 2010), np.nan)
        n_2010   = r.get(("total_inventors", 2010), np.nan)
        n_2020   = r.get(("total_inventors", 2020), np.nan)
        if np.isnan(hhi_2010) or np.isnan(n_2010) or np.isnan(n_2020):
            continue
        if n_2010 < 500:
            continue
        # Growth rate: log ratio so it's symmetric
        log_growth = np.log(max(n_2020, 1) / max(n_2010, 1))
        rows.append({
            "cpc_class": cls,
            "hhi_2010s": hhi_2010,
            "n_2010s": n_2010,
            "n_2020s": n_2020,
            "log_growth_inv_count": log_growth,
        })
    df = pd.DataFrame(rows).dropna()
    df.to_csv(OUT / "concentration_vs_innovation_pv.csv", index=False)
    return df


def main() -> int:
    tm = tm_concentration_innovation()
    pv = pv_concentration_innovation()

    # Correlations
    pe_tm, pp_tm = stats.pearsonr(tm["concentration"], tm["mean_dkl"])
    sr_tm, sp_tm = stats.spearmanr(tm["concentration"], tm["mean_dkl"])
    pe_pv, pp_pv = stats.pearsonr(pv["hhi_2010s"], pv["log_growth_inv_count"])
    sr_pv, sp_pv = stats.spearmanr(pv["hhi_2010s"], pv["log_growth_inv_count"])

    print("=== TRADEMARK: ethnic concentration vs mean ΔKL (n=%d classes)" % len(tm))
    print(f"  Pearson  r = {pe_tm:+.4f}   p = {pp_tm:.4f}")
    print(f"  Spearman r = {sr_tm:+.4f}   p = {sp_tm:.4f}")
    print()
    print("=== PATENT:    inventor-surname HHI(2010s) vs log(n_2020/n_2010) (n=%d classes)" % len(pv))
    print(f"  Pearson  r = {pe_pv:+.4f}   p = {pp_pv:.4f}")
    print(f"  Spearman r = {sr_pv:+.4f}   p = {sp_pv:.4f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    # Trademark panel
    ax = axes[0]
    colors = {"White":"#2b6cb0","API":"#229922","Hispanic":"#cc7a00","Black":"#cc4444"}
    for eth, group in tm.groupby("dominant_eth"):
        ax.scatter(group["concentration"], group["mean_dkl"],
                   s=np.clip(group["n_eff_total"]/30, 20, 250),
                   c=colors.get(eth, "#999"), alpha=0.7,
                   edgecolor="black", lw=0.4, label=f"dominant: {eth}")
    # Label most-concentrated classes (top 8 by concentration)
    for _, r in tm.head(8).iterrows():
        ax.annotate(f"{r['name']}", (r["concentration"], r["mean_dkl"]),
                    xytext=(5,5), textcoords="offset points", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white",
                              ec="#888", lw=0.5, alpha=0.8))
    # Fit line
    xs = np.linspace(tm["concentration"].min(), tm["concentration"].max(), 50)
    slope, intercept, *_ = stats.linregress(tm["concentration"], tm["mean_dkl"])
    ax.plot(xs, slope*xs + intercept, "--", color="#444", lw=1,
            label=f"OLS slope={slope:+.3f}\nPearson r={pe_tm:+.3f} (p={pp_tm:.3f})")
    ax.axhline(0, color="#888", lw=0.5)
    ax.set_xlabel("Ethnic concentration (1 − normalised entropy over 4 groups)")
    ax.set_ylabel(r"Mean $\Delta KL$ (signed; positive = innovation)")
    ax.set_title("Trademark side: NICE class concentration vs innovation")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    # Patent panel
    ax = axes[1]
    ax.scatter(pv["hhi_2010s"]*1000, pv["log_growth_inv_count"],
               s=np.clip(pv["n_2010s"]/100, 15, 200),
               c="#666", alpha=0.7, edgecolor="black", lw=0.3)
    # Label classes with rapid growth and high HHI together
    pv_sorted = pv.sort_values("hhi_2010s", ascending=False).head(8)
    for _, r in pv_sorted.iterrows():
        ax.annotate(r["cpc_class"], (r["hhi_2010s"]*1000, r["log_growth_inv_count"]),
                    xytext=(5,3), textcoords="offset points", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="#888", lw=0.4, alpha=0.8))
    slope_p, intercept_p, *_ = stats.linregress(pv["hhi_2010s"]*1000, pv["log_growth_inv_count"])
    xs = np.linspace((pv["hhi_2010s"]*1000).min(), (pv["hhi_2010s"]*1000).max(), 50)
    ax.plot(xs, slope_p*xs + intercept_p, "--", color="#444", lw=1,
            label=f"OLS slope={slope_p:+.3f}\nPearson r={pe_pv:+.3f} (p={pp_pv:.3f})")
    ax.axhline(0, color="#888", lw=0.5)
    ax.set_xlabel(r"Inventor-surname HHI (2010s), $\times 10^3$")
    ax.set_ylabel(r"$\log(n_{2020s}\,/\,n_{2010s})$ -- log growth in inventor count")
    ax.set_title("Patent side: CPC class concentration vs growth in inventors")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    fig.suptitle("Does cultural concentration help or hurt innovation? Two correlation tests",
                 fontsize=12, y=1.00)
    fig.tight_layout()
    fig.savefig(OUT / "concentration_vs_innovation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
