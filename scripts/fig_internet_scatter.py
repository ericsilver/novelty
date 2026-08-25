"""Registration and the five-year proof disagree about internet-bearing filings.

One point per Nice class from internet_breakout.json: x is the
internet-minus-rest gap in registration rate, y the internet-minus-rest gap
in failure at the five-year proof of continued use. Colored by class group
(technology, other services, industrial goods, consumer goods). Classes with
fewer than 500 internet-bearing registrations are dropped, matching the
breakout table.

Output: paper/results/fig_internet_scatter.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "paper" / "results"

TECH = {"009", "035", "038", "042"}
SERVICES = {f"{i:03d}" for i in range(35, 46)} - TECH
INDUSTRIAL = {"001", "004", "006", "007", "009", "010", "011", "012", "017", "019"} - TECH
GROUPS = [("technology", "#2b6cb0"), ("other services", "#e67e22"),
          ("industrial goods", "#c0392b"), ("consumer goods", "#16a085")]


def group_of(cls: str) -> str:
    if cls in TECH:
        return "technology"
    if cls in SERVICES:
        return "other services"
    if cls in INDUSTRIAL:
        return "industrial goods"
    return "consumer goods"


def main() -> int:
    j = json.loads((RES / "internet_breakout.json").read_text())
    fig, ax = plt.subplots(figsize=(7.6, 6.2))
    colors = dict(GROUPS)
    seen = set()
    for cls, rec in sorted(j["per_class"].items()):
        w, r = rec["web"], rec["rest"]
        if (not w.get("gate_failure") or not r.get("gate_failure")
                or w["gate_failure"]["n"] < 500):
            continue
        x = 100 * (w["registration"]["rate"] - r["registration"]["rate"])
        y = 100 * (w["gate_failure"]["rate"] - r["gate_failure"]["rate"])
        grp = group_of(cls)
        ax.scatter([x], [y], s=42, color=colors[grp], alpha=0.9,
                   label=grp if grp not in seen else None)
        seen.add(grp)
        ax.annotate(rec["name"], (x, y), xytext=(4, 3), textcoords="offset points",
                    fontsize=6.6, color="#444")
    ax.axhline(0, color="#333", lw=0.9)
    ax.axvline(0, color="#333", lw=0.9)
    ax.set_xlabel("registration rate: internet-bearing minus rest of class (pp)")
    ax.set_ylabel("failure at the five-year proof: internet-bearing minus rest (pp)")
    ax.legend(fontsize=8.5, frameon=False, loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RES / "fig_internet_scatter.png", dpi=150, bbox_inches="tight")
    print("[done] fig_internet_scatter.png", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
