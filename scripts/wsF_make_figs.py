"""Figures for the construct-validity note."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
PUB = REPO / "data_publish"
RES = REPO / "paper" / "results"


def fig_leadlag():
    j = json.loads((RES / "wsC_within_firm_patents.json").read_text())
    ll = j["fe_leadlag"]
    keys = ["logpat_lag", "logpat", "logpat_lead"]
    labels = ["patents$_{t-1}$", "patents$_{t}$", "patents$_{t+1}$"]
    coefs = [ll[k]["coef"] for k in keys]
    ses = [ll[k]["se"] for k in keys]
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    x = np.arange(3)
    ax.bar(x, coefs, yerr=[1.96 * s for s in ses], capsize=5,
           color=["#b2182b" if c < 0 else "#2166ac" for c in coefs], alpha=.85)
    ax.axhline(0, color="k", lw=.8)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel(r"$\Delta$KL response (per $\sigma$)")
    ax.set_title("Within-firm: trademark $\\Delta$KL vs. patent timing\n(firm + year FE, firm-clustered 95% CI)")
    fig.tight_layout(); fig.savefig(RES / "wsC_leadlag.png", dpi=150); plt.close(fig)
    print("wrote wsC_leadlag.png")


def fig_discriminant():
    fy = pd.read_csv(PUB / "firm_year_dkl.csv")
    fy["cik"] = fy.cik.astype(str)
    w = fy.n_filings.clip(lower=1)
    fy["_wd"] = fy.mean_dkl * w
    firm = fy.groupby("cik").apply(
        lambda g: g._wd.sum() / g.n_filings.clip(lower=1).sum(), include_groups=False)
    pop = firm.values

    xw = pd.read_csv(PUB / "comparators" / "external_crosswalk.csv")
    xw["matched_firm_mean_dkl"] = pd.to_numeric(xw.matched_firm_mean_dkl, errors="coerce")
    m = xw[xw.matched_firm_mean_dkl.notna()]
    bcg = m[m.comparator.str.startswith("BCG")].matched_firm_mean_dkl.mean()
    mit = m[m.comparator.str.startswith("MIT")].matched_firm_mean_dkl.mean()

    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.hist(np.clip(pop, -0.6, 0.6), bins=60, color="#bbbbbb", alpha=.8,
            label=f"all public firms (n={len(pop):,})")
    ax.axvline(np.mean(pop), color="k", ls=":", lw=1.2, label=f"pop. mean ({np.mean(pop):+.3f})")
    ax.axvline(bcg, color="#2166ac", lw=2, label=f"BCG list mean ({bcg:+.3f})")
    ax.axvline(mit, color="#b2182b", lw=2, label=f"MIT TR50 mean ({mit:+.3f})")
    ax.set_xlabel(r"firm mean $\Delta$KL"); ax.set_ylabel("firms")
    ax.set_title("Discriminant validity: expert-rated innovators\nsit above the population on $\\Delta$KL")
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout(); fig.savefig(RES / "wsD_discriminant.png", dpi=150); plt.close(fig)
    print("wrote wsD_discriminant.png")


if __name__ == "__main__":
    fig_leadlag()
    fig_discriminant()
