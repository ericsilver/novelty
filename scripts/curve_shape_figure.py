"""The three gates, both axes, binned profiles with the linear and best fits.

Reads the profiles and fitted coefficients written by gate_curve_shapes.py and
draws them, so the figure and the model selection cannot drift apart. Each panel
shows the 40 equal-count bin means, the linear fit the paper's quintile
contrasts and per-sigma slopes implicitly assume, and the shape AIC selected
where that is not the linear one.

Output: paper/results/curve_shapes.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "paper" / "results"

# One categorical pair, assigned by role and never recycled.
C_POINT = "#2b6cb0"     # the data
C_LIN = "#a0aec0"       # the linear reading
C_BEST = "#c05621"      # the selected shape

PANELS = [
    ("registration", None, "A", "Registration", "Atypicality"),
    ("registration", None, "L", "Registration", "Lead"),
    ("survival", None, "A", "Use-proof gate failure", "Atypicality"),
    ("survival", None, "L", "Use-proof gate failure", "Lead"),
    ("commercial", "listed", "L", "Owner later listed", "Lead"),
    ("commercial", "reporting_not_listed", "L",
     "Owner SEC-reporting, not listed", "Lead"),
]


def f_lin(x, a, b):
    return a + b * x


def f_quad(x, a, b, c):
    return a + b * x + c * x * x


def f_exp(x, a, b, c):
    return a + b * np.exp(np.clip(c * x, -50, 50))


def f_sym(x, a, b, c, x0):
    return a + b * np.exp(np.clip(c * np.abs(x - x0), -50, 50))


FN = {"linear": f_lin, "quadratic": f_quad,
      "exponential": f_exp, "symmetric": f_sym}


def main() -> int:
    src = RES / "gate_curve_shapes.json"
    if not src.exists():
        print("run scripts/gate_curve_shapes.py first", file=sys.stderr)
        return 1
    d = json.load(src.open())

    fig, axes = plt.subplots(3, 2, figsize=(10.5, 10.2))
    for ax, (gate, tier, axis, title, xlab) in zip(axes.ravel(), PANELS):
        node = d["gates"][gate]
        if tier:
            node = node[tier]
        p = node[axis]
        if not p:
            ax.set_visible(False)
            continue
        x = np.array(p["bins"]["x_z"])
        y = np.array(p["bins"]["y"]) * 100
        ax.scatter(x, y, s=16, color=C_POINT, zorder=3, label="bin means")

        xs = np.linspace(x.min(), x.max(), 300)
        fits = p["fits"]
        lin = fits.get("linear", {})
        if "params" in lin:
            ax.plot(xs, f_lin(xs, *lin["params"]) * 100, color=C_LIN, lw=2.0,
                    zorder=2, label=f"linear ($R^2$={lin['r2']:.2f})")
        best = fits.get("_best")
        if best and best != "linear" and "params" in fits[best]:
            ax.plot(xs, FN[best](xs, *fits[best]["params"]) * 100,
                    color=C_BEST, lw=2.0, zorder=4,
                    label=f"{best} ($R^2$={fits[best]['r2']:.2f})")
            v = fits.get("_quadratic_vertex_z")
            if best == "quadratic" and v is not None and x.min() < v < x.max():
                ax.axvline(v, color=C_BEST, lw=1.0, ls=":", zorder=1)
        ax.set_title(title, fontsize=10.5, loc="left")
        ax.set_xlabel(f"{xlab} (standard deviations)", fontsize=9)
        ax.set_ylabel("Rate (%)", fontsize=9)
        ax.legend(loc="best", frameon=False, fontsize=8)
        ax.grid(axis="y", color="#e2e8f0", lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#a0aec0")
        ax.tick_params(labelsize=8)

    fig.tight_layout()
    fig.savefig(RES / "curve_shapes.png", dpi=200)
    plt.close(fig)
    print("[done] curve_shapes.png", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
