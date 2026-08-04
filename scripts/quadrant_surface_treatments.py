"""Three statistical treatments of the quadrant outcome rate surfaces.

THE PROBLEM.  quadrant_survival.png and quadrant_listing.png (built by
scripts/quadrant_outcome_variants.py) colour each hex cell by an outcome rate
measured against that panel's own base rate.  Cell denominators are wildly
heterogeneous -- on the class-009 survival surface the smallest drawn cell holds
85 registrations and the largest 6,150 -- so a marginal cell paints a large part
of the colour ramp on sampling noise alone.  At n=85 and a 46.2% base the
per-cell standard error is 5.4pp against a +/-16pp ramp; at n=40 (class 032) it
is 7.9pp.  The dense core is trustworthy, the fringe is fireworks.  This is the
standard disease-mapping problem: rates on a grid with varying denominators.

FILINGS ARE NOT INDEPENDENT DRAWS.  Every standard error here is computed on an
effective cell size n/deff, not on the filing count, where deff = 1 + (mbar-1)*rho
is Kish's design effect for owner clustering: mbar is the cell's cluster-size-
weighted mean owner size and rho the intraclass correlation of the outcome within
owner.  On the listing surface rho is exactly 1 -- "listed" is an attribute of the
owner, so a firm with 300 filings in a cell contributes 300 identical successes,
not 300 draws -- and the median cell design effect is 1.75 in class 009 and 2.61
in class 032.  On the survival surface rho is about 0.42 and the median design
effect is 1.27 and 1.37.  Correcting for this changes the headline diagnostic
substantially: the excess of cell-to-cell spread over sampling noise falls from
3.48x to 2.8x and 2.31x to 1.9x on the survival surface, and from 2.11x to 1.3x
and 2.79x to 1.6x on the listing surface.  The survival surface still carries
plainly real structure.  The class-009 listing surface has almost none left.

Given that, this is a display problem on the survival surface -- the structure is
real and the question is only how much of any individual cell's colour to believe
-- and partly a "there is less here than it looks" problem on the listing surface.

THREE TREATMENTS, each applied to both surfaces and both class panels.

  EB   quadrant_{survival,listing}_eb.{png,pdf}
       Empirical-Bayes (Clayton-Kaldor / beta-binomial) shrinkage.  A beta
       prior is estimated from the observed spread of cell rates by Kleinman's
       weighted method of moments, and each cell's rate is replaced by its
       posterior mean, which is a precision-weighted blend of the cell's own
       rate and the prior mean.  Small cells move most.  Nothing is hidden and
       the map stays complete; the surface simply stops asserting more than the
       denominators support.

  SM   quadrant_{survival,listing}_smooth.{png,pdf}
       Kernel-regression ("topographic") surface.  Nadaraya-Watson regression
       of the binary outcome on (kl_vs_past, kl_vs_future) with a Gaussian
       kernel, computed by binning to a 240x240 lattice and convolving, which
       is exactly kernel regression evaluated on binned data.  A logistic GAM
       with a tensor-product spline was the first choice but neither pygam nor
       an interaction-capable statsmodels GAM is importable in this venv, and
       statsmodels' GLMGam with BSplines is additive, which cannot represent an
       off-diagonal ridge; kernel regression is the stated fallback.  The
       bandwidth is chosen by 5-fold cross-validation on out-of-fold binomial
       deviance over sigma in [0.04, 1.10] (see CV_SIGMAS), with the folds
       BLOCKED ON OWNER; a paired one-standard-error rule is reported as a
       robustness check.  Blocking is not a detail: under point-level folds the
       selected bandwidth is 0.06 everywhere, because 22-35% of filings share an
       exact coordinate with another filing and the listing outcome is an owner
       attribute, so the fit is rewarded for memorising positions it has already
       seen.  Blocking on owner raises the selected bandwidth to 0.08 and 0.25 on
       the survival surface and to 0.65 and 0.25 on the listing surface.
       Contours borrow strength from neighbours, but they can also imply
       precision that is not there, so the region where a pointwise 95% band
       covers the panel base rate is veiled and hatched.  The band is
       variance-only: it does not carry the smoothing bias, which is the usual
       caveat on a pointwise kernel-regression band.  The surface is drawn only
       where the hexbin figures are drawn -- a wide kernel meets any n_eff floor
       far outside the data (at sigma=0.65 over three quarters of the box,
       corners included), and a fitted value there is extrapolation.

  SIG  quadrant_{survival,listing}_sig.{png,pdf}
       The current surface with cells whose deviation from base is under ~2
       standard errors drawn in neutral.  With 282 drawn cells on the class-009
       survival panel, about 13 would clear |z|>2 by chance at the 5% level, so
       a Benjamini-Hochberg step-up at q=0.05 is applied as well: cells that
       clear |z|>2 but do NOT survive BH are hatched.  Both counts are printed
       on the panel.

WHAT IS HELD FIXED.  Axes, sample definitions, exemplar keys and the completeness
guard are inherited unchanged from quadrant_outcome_variants.py, which this
module imports rather than copies.  Colour semantics are unchanged: BLUE is the
better commercial outcome on both surfaces, which is why the listing surface
uses the reversed ramp.  Colourbars stay per-panel and anchored to each panel's
own base rate.  Exemplars are keyed on (mark, year, owner) and any filing
without a complete five-year forward window is dropped, not marked.

Outputs: paper/results/quadrant_survival_{eb,smooth,sig}.{png,pdf}
         paper/results/quadrant_listing_{eb,smooth,sig}.{png,pdf}
         paper/results/quadrant_surface_treatments.csv
         paper/results/quadrant_surface_treatment_cells.csv
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from scipy.stats import norm as NORM

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "paper" / "results"
sys.path.insert(0, str(REPO / "scripts"))

import quadrant_outcome_variants as Q  # noqa: E402
from quadrant_outcome_variants import (  # noqa: E402
    BLUE, ORANGE, INK, INK2, MUTED, GRID, DIVERGING, DIVERGING_R,
    PANELS, LO, HI, SURF_LO, SURF_HI, HEX_SURV, HEX_LIST, SURV_PP, LIST_L2,
    Z_MARK, find_exemplar, label_exemplar, place_labels, style_axes, log,
)

NEUTRAL = "#d5d3cb"      # "not distinguishable from base" fill
NB = 240                 # smoothing lattice per side over the [2,10] box
CV_SIGMAS = np.array([0.04, 0.06, 0.08, 0.10, 0.13, 0.16, 0.20, 0.25, 0.32,
                      0.40, 0.50, 0.65, 0.85, 1.10])
CV_FOLDS = 5
CV_SEED = 11
BANDS = 8                # colour bands used to count "did the cell change colour"
FDR_Q = 0.05
matplotlib.rcParams["hatch.linewidth"] = 0.65


# --------------------------------------------------------------- surfaces ---
class Surface:
    """One of the two rate surfaces: its sample, outcome, colour map and scale."""

    def __init__(self, name):
        self.name = name
        if name == "survival":
            self.cmap = DIVERGING
            self.rng = SURV_PP
            self.hex = HEX_SURV
            self.lines = [-8, -4, 4, 8]
            self.unit = "pp"
        else:
            self.cmap = DIVERGING_R
            self.rng = LIST_L2
            self.hex = HEX_LIST
            self.lines = [-1.0, -0.5, 0.5, 1.0]
            self.unit = "log2"

    def make_norm(self):
        """A FRESH Normalize per panel.

        Two colourbars sharing one Normalize instance share its callback list, so
        building the second one fires update_normal() on the first and silently
        resets its fixed ticks back to the automatic locator.
        """
        return Normalize(vmin=-self.rng, vmax=self.rng)

    def sample(self, d: pl.DataFrame):
        """(x, y, binary outcome, dKL) for one panel, on the surface's own sample."""
        if self.name == "survival":
            s = d.filter(pl.col("gate_elapsed"))
            f = s["failed1"].to_numpy().astype(float)
        else:
            s = d.filter(pl.col("year").is_between(SURF_LO, SURF_HI)
                         & pl.col("reg_d").is_not_null())
            f = s["is_pub"].to_numpy().astype(float)
        return (s["kl_vs_past"].to_numpy(), s["kl_vs_future"].to_numpy(), f,
                s["dkl"].to_numpy(), s)

    def value(self, p, base):
        """Map a rate onto the plotted colour variable (pp, or log2 of the ratio)."""
        p = np.asarray(p, dtype=float)
        if self.name == "survival":
            return 100.0 * (p - base)
        return np.log2(np.maximum(p, base / 64.0) / base)

    def cbar_label(self, base, extra):
        if self.name == "survival":
            return (f"first-gate failure rate, percentage points\n"
                    f"from this panel's base of {100*base:.1f}% {extra}\n"
                    f"blue = more likely to survive")
        return (f"listing rate as a multiple of this\n"
                f"panel's base of {100*base:.2f}% (log scale) {extra}\n"
                f"blue = more likely to be listed")

    def cbar_ticks(self, cb):
        if self.name == "survival":
            cb.set_ticks([-16, -8, 0, 8, 16])
            cb.set_ticklabels(["$-$16", "$-$8", "base", "+8", "+16"])
        else:
            cb.set_ticks([-2.0, -1.0, 0.0, 1.0, 2.0])
            cb.set_ticklabels(["0.25$\\times$", "0.5$\\times$", "base",
                               "2$\\times$", "4$\\times$"])
        cb.ax.tick_params(labelsize=7, colors=INK2)
        cb.outline.set_edgecolor(GRID)

    def exemplar_spec(self, r):
        if self.name == "survival":
            if not r["gate_elapsed"]:
                return "not classifiable", dict(marker="^", mfc="white", mec=MUTED,
                                                ms=7.0)
            if r["failed1"]:
                return "failed", dict(marker="s", mfc="none", mec=ORANGE, ms=8.0)
            return "survived", dict(marker="o", mfc=BLUE, mec="white", ms=8.0)
        if r["is_pub"]:
            return "owner listed", dict(marker="o", mfc=INK, mec="white", ms=7.5)
        return "owner not listed", dict(marker="D", mfc="none", mec=INK, ms=6.5)

    def legend(self):
        if self.name == "survival":
            return [
                Line2D([], [], linestyle="none", marker="o", mfc=BLUE, mec="white",
                       ms=8, mew=1.5, label="named mark: survived the gate"),
                Line2D([], [], linestyle="none", marker="s", mfc="none", mec=ORANGE,
                       ms=8, mew=1.5, label="named mark: failed the gate"),
                Line2D([], [], linestyle="none", marker="^", mfc="white", mec=MUTED,
                       ms=7, mew=1.5, label="named mark: not classifiable "
                       "(unregistered or gate not yet elapsed)"),
            ]
        return [
            Line2D([], [], linestyle="none", marker="o", mfc=INK, mec="white",
                   ms=7.5, mew=1.4, label="named mark: owner in the listed universe"),
            Line2D([], [], linestyle="none", marker="D", mfc="none", mec=INK,
                   ms=6.5, mew=1.4, label="named mark: owner not in the listed universe"),
        ]


# ------------------------------------------------------------- hex binning --
def hex_metric(gs):
    """Coordinate scaling under which matplotlib's hexagons are regular.

    hexbin bins on nx = gridsize columns and ny = int(nx/sqrt(3)) rows and picks
    the nearer of two candidate centres under (dix)^2 + 3*(diy)^2 measured in
    grid-index units.  Because the plotting box is square but nx != ny, that
    metric is NOT Euclidean distance in data units; a nearest-centre lookup done
    in raw data coordinates assigns a few percent of points to the wrong cell.
    """
    nx = gs
    ny = int(nx / np.sqrt(3.0))
    return (HI - LO) / nx, (HI - LO) / ny / np.sqrt(3.0)


def hex_lattice(ax, x, y, gs):
    """Cell centres and a point-to-cell index for every non-empty cell.

    The hexagons are the Voronoi cells of their centres under the metric above,
    so a nearest-centre lookup reproduces matplotlib's own assignment exactly.
    Doing it this way -- rather than through two reduce_C_function passes -- means
    the counts, the outcome sums and the drawn geometry cannot silently disagree.
    """
    hb = ax.hexbin(x, y, gridsize=gs, extent=(LO, HI, LO, HI), mincnt=1,
                   visible=False)
    off = np.array(hb.get_offsets())
    hb.remove()
    ax_, ay_ = hex_metric(gs)
    idx = cKDTree(np.column_stack([off[:, 0] / ax_, off[:, 1] / ay_])).query(
        np.column_stack([x / ax_, y / ay_]))[1]
    return off, idx


def cell_stats(idx, f, ncell):
    n = np.bincount(idx, minlength=ncell).astype(float)
    s = np.bincount(idx, weights=f, minlength=ncell).astype(float)
    return n, s


def intra_owner_correlation(oid, f):
    """One-way ANOVA estimate of the intraclass correlation of the outcome by owner.

    On the listing surface this is exactly 1: "listed" is an attribute of the
    owner, so every filing by the same firm carries the same value and the within-
    owner sum of squares is zero.  On the survival surface it is well below 1 but
    well above 0, because a firm that lets one registration lapse tends to let its
    siblings lapse too.
    """
    N = len(f)
    cnt = np.bincount(oid).astype(float)
    tot = np.bincount(oid, weights=f)
    G = len(cnt)
    gbar = tot / cnt
    ybar = f.mean()
    ssb = float((cnt * (gbar - ybar) ** 2).sum())
    ssw = float(((f - gbar[oid]) ** 2).sum())
    if G <= 1 or N <= G:
        return 0.0
    msb, msw = ssb / (G - 1), ssw / (N - G)
    m0 = (N - (cnt ** 2).sum() / N) / (G - 1)
    den = msb + (m0 - 1) * msw
    return float(np.clip((msb - msw) / den, 0.0, 1.0)) if den > 0 else 0.0


def owner_design_effect(idx, oid, ncell, rho):
    """Per-cell design effect from owner clustering, in Kish's form.

    Filings by the same owner do not carry independent information.  On the
    listing surface they carry none at all beyond the first, because "listed" is
    an owner attribute: a firm with 300 filings in one cell contributes 300
    identical successes, not 300 draws.  The standard correction is
        deff_i = 1 + (mbar_i - 1) * rho,     mbar_i = sum_g n_ig^2 / n_i
    where mbar is the cell's cluster-size-weighted mean owner size and rho is the
    intraclass correlation.  mbar is exact per cell, rho is estimated once on the
    whole panel, so the correction varies with the cell's own ownership mix
    without importing the estimation noise of a per-cell variance estimate.  At
    rho = 1 this reduces to the exact effective sample size n_i / mbar_i.
    Every standard error in this module uses n_i / deff_i, not n_i.
    """
    key = idx.astype(np.int64) * (int(oid.max()) + 1) + oid
    _, kinv, gn = np.unique(key, return_inverse=True, return_counts=True)
    gcell = np.zeros(len(gn), dtype=np.int64)
    gcell[kinv] = idx                                 # every member shares the cell
    n = np.bincount(idx, minlength=ncell).astype(float)
    sq = np.bincount(gcell, weights=gn.astype(float) ** 2, minlength=ncell)
    mbar = np.divide(sq, n, out=np.ones(ncell), where=n > 0)
    return np.maximum(1.0 + (mbar - 1.0) * rho, 1.0), mbar


def in_drawn_cells(X, Y, off, sel, gs):
    """Which points of the smoothing lattice fall inside a drawn hexbin cell."""
    ax_, ay_ = hex_metric(gs)
    who = cKDTree(np.column_stack([off[:, 0] / ax_, off[:, 1] / ay_])).query(
        np.column_stack([X.ravel() / ax_, Y.ravel() / ay_]))[1]
    return sel[who].reshape(X.shape)


def deff_at_scale(ax, x, y, oid, rho, sigma, floor=30):
    """Design effect for a window of the same effective area as the kernel.

    Clustering is scale-dependent: a wider window captures more of each owner's
    portfolio, so its design effect is larger.  The hexbin figures can use their
    own cells, but the smoothed surface needs the design effect at the bandwidth
    it actually uses.  A Gaussian kernel of width sigma has effective area
    4*pi*sigma^2, and a hexbin of gridsize g over this box has cells of area
    about 64*sqrt(3)/g^2, so the two match at g ~ 2.97/sigma; the design effect is
    read off a lattice at that gridsize.
    """
    gs = int(np.clip(round(2.97 / max(sigma, 1e-3)), 3, 60))
    off, idx = hex_lattice(ax, x, y, gs)
    deff, _ = owner_design_effect(idx, oid, len(off), rho)
    n = np.bincount(idx, minlength=len(off)).astype(float)
    keep = n >= floor
    return float(np.average(deff[keep], weights=n[keep])), gs


def align(off, got):
    """Map the cells a drawn hexbin emits back onto our lattice indices.

    matplotlib emits cells in a different order depending on whether C is given,
    so the drawn collection is matched to the lattice on geometry rather than on
    position in the array.
    """
    dist, perm = cKDTree(off).query(got)
    assert dist.max() < 1e-6, f"drawn cell not on the lattice (max gap {dist.max()})"
    assert len(np.unique(perm)) == len(perm), "drawn cells collide on the lattice"
    return perm


def draw_cells(ax, x, y, gs, mincnt, off, sel, values, cmap, norm):
    """Draw the hexbin lattice and substitute our own per-cell values."""
    hb = ax.hexbin(x, y, C=np.zeros(len(x)), gridsize=gs, extent=(LO, HI, LO, HI),
                   mincnt=mincnt, reduce_C_function=np.mean, cmap=cmap, norm=norm,
                   linewidths=0.0, zorder=1)
    got = np.array(hb.get_offsets())
    assert len(got) == int(sel.sum()), (len(got), int(sel.sum()))
    perm = align(off, got)
    assert sel[perm].all(), "hexbin drew a cell below the minimum count"
    hb.set_array(np.ma.masked_invalid(values[perm]))
    return hb


def hatch_cells(ax, x, y, idx, gs, cells):
    """Overlay a hatch on the given lattice cells, drawn on the same lattice."""
    m = np.isin(idx, np.where(cells)[0])
    if not m.any():
        return None
    hb = ax.hexbin(x[m], y[m], gridsize=gs, extent=(LO, HI, LO, HI), mincnt=1,
                   linewidths=0.0, zorder=2)
    hb.set_facecolor("none")
    hb.set_edgecolor(INK2)
    hb.set_hatch("///")
    hb.set_linewidth(0.0)
    return hb


# ------------------------------------------------------ empirical Bayes -----
def eb_prior(n, y):
    """Beta prior by Kleinman's weighted method of moments (Clayton-Kaldor family).

    Returns (prior mean m, prior variance s2, prior sample size nu).  The
    m(1-m)(K-1) term removes the binomial component of the observed spread, so
    s2 estimates only the between-cell variance; nu = m(1-m)/s2 - 1 is the beta
    prior's equivalent sample size, and a cell of size n is pulled a fraction
    nu/(n+nu) of the way to the prior mean.
    """
    N, K = n.sum(), len(n)
    m = y.sum() / N
    r = y / n
    num = (n * (r - m) ** 2).sum() - m * (1 - m) * (K - 1)
    den = N - (n ** 2).sum() / N
    s2 = float(max(num / den, 1e-12))
    nu = float(m * (1 - m) / s2 - 1.0)
    return float(m), s2, nu


def bh_reject(p, q=FDR_Q):
    """Benjamini-Hochberg step-up; returns the rejection mask."""
    K = len(p)
    order = np.argsort(p)
    passed = p[order] <= q * np.arange(1, K + 1) / K
    out = np.zeros(K, bool)
    if passed.any():
        out[order[: int(np.max(np.where(passed)[0])) + 1]] = True
    return out


# ------------------------------------------------------- kernel regression --
def bin2d(x, y, f):
    h = (HI - LO) / NB
    ix = np.clip(((x - LO) / h).astype(int), 0, NB - 1)
    iy = np.clip(((y - LO) / h).astype(int), 0, NB - 1)
    flat = ix * NB + iy
    c = np.bincount(flat, minlength=NB * NB).astype(float).reshape(NB, NB)
    s = np.bincount(flat, weights=f, minlength=NB * NB).astype(float).reshape(NB, NB)
    return c, s, ix, iy


def nw(c, s, sig_pix):
    """Nadaraya-Watson fit and effective sample size on the binned lattice.

    With an unnormalized Gaussian kernel K, sum_i K_i and sum_i K_i^2 are both
    convolutions of the binned counts -- the second with the kernel squared,
    which is Gaussian at sigma/sqrt(2) -- so
        n_eff = (sum K)^2 / sum K^2 = 4 pi sigma^2 F1^2 / F2
    with F1, F2 the two normalized-Gaussian filters of the count field.  n_eff is
    the denominator the pointwise standard error should use: it is the number of
    observations the kernel is actually averaging, not the number in a bin.
    """
    kw = dict(mode="constant", cval=0.0, truncate=5.0)
    F1c = gaussian_filter(c, sig_pix, **kw)
    F1y = gaussian_filter(s, sig_pix, **kw)
    F2c = gaussian_filter(c, sig_pix / np.sqrt(2.0), **kw)
    ok = (F1c > 1e-9) & (F2c > 1e-9)
    p = np.where(ok, F1y / np.where(ok, F1c, 1.0), np.nan)
    neff = np.where(ok, 4 * np.pi * sig_pix ** 2 * F1c ** 2 / np.where(ok, F2c, 1.0),
                    0.0)
    return p, neff


def cv_bandwidth(x, y, f, oid):
    """5-fold CV on out-of-fold binomial deviance, with folds blocked on owner.

    Blocking matters more than the choice of criterion here.  Under point-level
    folds the same owner, and often the very same (kl_vs_past, kl_vs_future)
    coordinate, appears on both sides of every split -- 22% of class-009 filings
    and 35% of class-032 filings sit on a coordinate shared with another filing --
    so the fit is rewarded for memorising locations it has already seen, and the
    selected bandwidth collapses toward the grid.  On the listing surface the
    leak is total, because the outcome is an owner attribute.  Holding out whole
    owners removes it; the selected bandwidth roughly doubles on the survival
    surface and rises tenfold on the class-009 listing surface.
    """
    h = (HI - LO) / NB
    rng = np.random.default_rng(CV_SEED)
    fold = rng.integers(0, CV_FOLDS, int(oid.max()) + 1)[oid]
    c_all, s_all, ix, iy = bin2d(x, y, f)
    parts = [bin2d(x[fold == j], y[fold == j], f[fold == j])[:2]
             for j in range(CV_FOLDS)]
    D = np.zeros((len(CV_SIGMAS), CV_FOLDS))
    for a, sg in enumerate(CV_SIGMAS):
        for j in range(CV_FOLDS):
            m = fold == j
            p, _ = nw(c_all - parts[j][0], s_all - parts[j][1], sg / h)
            p = np.clip(np.nan_to_num(p, nan=f.mean()), 1e-4, 1 - 1e-4)
            ph = p[ix[m], iy[m]]
            D[a, j] = -(f[m] * np.log(ph) + (1 - f[m]) * np.log(1 - ph)).mean()
    return D


# ------------------------------------------------------------- shared bits --
def quintile_contrast(dkl, rates):
    """Headline check: Q5 minus Q1 of a per-filing rate across dKL quintiles.

    Reported in percentage points on both surfaces, and as the Q5/Q1 ratio, so
    that a treatment's effect on the headline pattern is comparable across the
    two.  On the raw data this is +6.2pp in class 009 (below-diagonal marks fail
    the gate MORE) and -6.1pp in class 032 (the sign reverses).
    """
    qi = np.searchsorted(np.quantile(dkl, [.2, .4, .6, .8]), dkl)
    r = np.array([np.nanmean(rates[qi == q]) for q in range(5)])
    return 100.0 * r, 100.0 * float(r[4] - r[0]), float(r[4] / r[0])


def band(v, rng):
    """Colour-band index of a plotted value, on BANDS equal bands across the ramp."""
    w = 2.0 * rng / BANDS
    return np.clip(np.floor((np.clip(v, -rng, rng) + rng) / w), 0, BANDS - 1)


def add_exemplars(ax, surf, d):
    xs, ys, texts, rows = [], [], [], []
    for mark, year, owner in d["_examples"]:
        r = find_exemplar(d["frame"], mark, year, owner)
        if r is None:
            continue
        state, spec = surf.exemplar_spec(r)
        px, py, _ = label_exemplar(ax, r, mark, year, xs, ys, texts)
        ax.plot([px], [py], linestyle="none", mew=1.4 if surf.name == "listing"
                else 1.5, zorder=Z_MARK, **spec)
        rows.append((mark, year, state))
    place_labels(ax, texts, xs, ys)
    return rows


def finish(fig, surf, suptitle, stem):
    extra = []
    if stem.endswith("_sig"):
        extra = [Patch(facecolor=NEUTRAL, edgecolor="none",
                       label="cell not distinguishable from the panel base rate"),
                 Patch(facecolor="white", edgecolor=INK2, hatch="///",
                       label="cell clears $|z|>2$ but not the FDR cut")]
    elif stem.endswith("_smooth"):
        extra = [Patch(facecolor="white", edgecolor=MUTED, hatch="\\\\",
                       label="fitted surface not distinguishable from the base rate")]
    fig.subplots_adjust(bottom=0.21, top=0.80, wspace=0.42)
    fig.legend(handles=surf.legend() + extra, loc="lower center", ncol=2,
               frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 0.005))
    fig.suptitle(suptitle, fontsize=12.5, color=INK, y=0.985)
    for ext in ("png", "pdf"):
        fig.savefig(RES / f"{stem}.{ext}", dpi=170, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)
    log(f"[write] {stem}.png / .pdf")


HEAD = {"survival": "Vocabulary position and survival at the first Section 8 "
                    "use-proof gate",
        "listing": "Vocabulary position and the chance the owner reached the "
                   "listed universe"}
SUB = {"survival": "registrations 2002-2017, the cohorts whose first gate has "
                   "fully elapsed",
       "listing": "granted marks filed 1995-2020, the years with a complete "
                  "five-year forward window"}


# ================================================================ treatment 1
def figure_eb(surf, panels, diag, cells_out):
    fig, axes = plt.subplots(1, 2, figsize=(15.6, 6.4))
    fig.patch.set_facecolor("white")
    for ax, P in zip(axes, panels):
        x, y, f, dkl, base = P["x"], P["y"], P["f"], P["dkl"], P["base"]
        gs, mincnt = surf.hex[P["cls"]]
        off, idx, n, ysum, sel = P["off"], P["idx"], P["n"], P["ysum"], P["sel"]
        neff = P["neff"]
        # the prior is estimated on effective cell sizes, so the binomial part of
        # the observed spread is not understated and the shrinkage is not too weak
        m, s2, nu = eb_prior(neff[sel], (neff * np.divide(ysum, np.maximum(n, 1)))[sel])
        a, b = m * nu, (1 - m) * nu
        raw = np.full(len(n), np.nan)
        post = np.full(len(n), np.nan)
        raw[sel] = ysum[sel] / n[sel]
        post[sel] = (neff[sel] * raw[sel] + a) / (neff[sel] + a + b)
        w = nu / (neff + nu)                 # weight the posterior puts on the prior
        v_raw, v_post = surf.value(raw, base), surf.value(post, base)
        v_raw[~sel] = np.nan
        v_post[~sel] = np.nan

        hb = draw_cells(ax, x, y, gs, mincnt, off, sel, v_post, surf.cmap,
                        surf.make_norm())
        cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label(surf.cbar_label(base, "(EB-shrunk)"), fontsize=7.5, color=INK2)
        surf.cbar_ticks(cb)

        # how far the loudest cells move, and how many change colour band
        top = np.argsort(np.abs(v_raw[sel]))[-10:]
        mv_raw = float(np.mean(np.abs(v_raw[sel][top])))
        mv_post = float(np.mean(np.abs(v_post[sel][top])))
        changed = int((band(v_raw[sel], surf.rng) != band(v_post[sel], surf.rng)).sum())
        dist_raw = int((np.abs(v_raw[sel]) > surf.rng / 4).sum())
        dist_post = int((np.abs(v_post[sel]) > surf.rng / 4).sum())
        _, q_raw, qr_raw = quintile_contrast(dkl, f)
        _, q_post, qr_post = quintile_contrast(dkl, post[idx])
        expsd = P["noise_sd"]
        sd_post = float(np.nanstd(v_post[sel]))

        log(f"[EB/{surf.name}/{P['cls']}] prior mean {m:.5f} var {s2:.3e} "
            f"(sd {100*np.sqrt(s2):.2f}pp) nu {nu:.1f}; prior weight "
            f"min {w[sel].min():.3f} med {np.median(w[sel]):.3f} max {w[sel].max():.3f}; "
            f"loudest 10 cells |dev| {mv_raw:.2f} -> {mv_post:.2f} {surf.unit} "
            f"({100*(1-mv_post/mv_raw):.0f}% shrunk); {changed} of {int(sel.sum())} "
            f"cells change colour band; {dist_raw} -> {dist_post} cells beyond a "
            f"quarter-ramp; sd {P['raw_sd']:.3f} -> {sd_post:.3f} vs {expsd:.3f} "
            f"noise (dispersion {sd_post/expsd:.2f}); dKL Q5-Q1 {q_post:+.2f}pp "
            f"({qr_post:.2f}x) against {q_raw:+.2f}pp ({qr_raw:.2f}x) raw")

        style_axes(ax, f"{P['label']}  (class {P['cls']})",
                   f"{len(x):,} filings   |   base {100*base:.2f}%   |   "
                   f"{int(sel.sum())} cells   |   prior sd "
                   f"{100*np.sqrt(s2):.2f}pp, $\\nu$={nu:.0f}\n"
                   f"median cell pulled {100*np.median(w[sel]):.0f}% to the prior, "
                   f"the smallest {100*w[sel].max():.0f}%; raw spread "
                   f"{P['raw_sd']/expsd:.1f}$\\times$ clustered sampling noise")
        add_exemplars(ax, surf, P)
        diag.append(dict(surface=surf.name, cls=P["cls"], treatment="empirical_bayes",
                         n_points=len(x), base_rate=round(base, 5),
                         cells_drawn=int(sel.sum()),
                         intra_owner_correlation=round(P["rho"], 4),
                         owner_deff_median=round(float(np.median(P["deff"][sel])), 3),
                         owner_deff_max=round(float(P["deff"][sel].max()), 3),
                         prior_mean=round(m, 5), prior_var=float(f"{s2:.4g}"),
                         prior_sd_pp=round(100 * np.sqrt(s2), 3),
                         prior_nu=round(nu, 1),
                         shrink_weight_median=round(float(np.median(w[sel])), 3),
                         shrink_weight_max=round(float(w[sel].max()), 3),
                         top10_absdev_raw=round(mv_raw, 3),
                         top10_absdev_treated=round(mv_post, 3),
                         cells_changing_colour_band=changed,
                         cells_visually_distinct_raw=dist_raw,
                         cells_visually_distinct=dist_post,
                         plotted_sd_raw=round(P["raw_sd"], 3),
                         plotted_sd=round(sd_post, 3),
                         noise_sd_clustered=round(expsd, 3),
                         noise_sd_binomial=round(P["noise_sd_bin"], 3),
                         dispersion_raw=round(P["raw_sd"] / expsd, 2),
                         dispersion=round(sd_post / expsd, 2),
                         dkl_q5_minus_q1_pp_raw=round(q_raw, 3),
                         dkl_q5_minus_q1_pp=round(q_post, 3),
                         dkl_q5_over_q1_raw=round(qr_raw, 3),
                         dkl_q5_over_q1=round(qr_post, 3)))
        for i in np.where(sel)[0]:
            cells_out.append(dict(surface=surf.name, cls=P["cls"],
                                  x=round(float(off[i, 0]), 4),
                                  y=round(float(off[i, 1]), 4), n=int(n[i]),
                                  n_effective=round(float(neff[i]), 1),
                                  rate_raw=round(float(raw[i]), 5),
                                  rate_eb=round(float(post[i]), 5),
                                  value_raw=round(float(v_raw[i]), 4),
                                  value_eb=round(float(v_post[i]), 4),
                                  z=round(float(P["z"][i]), 3),
                                  p=float(f"{P['pv'][i]:.4g}"),
                                  sig_naive=bool(abs(P["z"][i]) > 2),
                                  sig_bh=bool(P["bh"][i])))
    finish(fig, surf, HEAD[surf.name] + "\n" + SUB[surf.name]
           + ";  cell rates shrunk toward an empirical-Bayes prior estimated "
             "from the spread of the cells themselves",
           f"quadrant_{surf.name}_eb")


# ================================================================ treatment 2
def figure_smooth(surf, panels, diag):
    fig, axes = plt.subplots(1, 2, figsize=(15.6, 6.4))
    fig.patch.set_facecolor("white")
    scratch, sax = plt.subplots()
    h = (HI - LO) / NB
    ctr = LO + h * (np.arange(NB) + 0.5)
    X, Y = np.meshgrid(ctr, ctr, indexing="ij")
    for ax, P in zip(axes, panels):
        x, y, f, dkl, base = P["x"], P["y"], P["f"], P["dkl"], P["base"]
        mincnt = surf.hex[P["cls"]][1]
        D = cv_bandwidth(x, y, f, P["oid"])
        mu = D.mean(1)
        i0 = int(mu.argmin())
        sed = (D - D[i0]).std(1, ddof=1) / np.sqrt(CV_FOLDS)
        i1 = int(np.max(np.where(mu - mu[i0] <= sed)[0]))
        sig = float(CV_SIGMAS[i0])
        deff, dgs = deff_at_scale(sax, x, y, P["oid"], P["rho"], sig)

        c, s, ix, iy = bin2d(x, y, f)
        p, nk = nw(c, s, sig / h)
        neff = nk / deff                   # kernel weight count, discounted for owners
        # Support is the region the hexbin figures actually draw, intersected with
        # the same information floor.  A wide kernel satisfies the floor a long way
        # outside the data -- at sigma=0.65 on the class-009 listing panel it does
        # so over three quarters of the box, including corners holding no filings
        # at all -- and a fitted value there is extrapolation, not smoothing.
        # Tying the domain to the hexbins also makes the three treatments cover
        # exactly the same ground, which is what makes them comparable.
        supp = in_drawn_cells(X, Y, P["off"], P["sel"], surf.hex[P["cls"]][0]) \
            & (neff >= mincnt)
        se = np.sqrt(np.clip(p * (1 - p), 1e-12, None) / np.clip(neff, 1.0, None))
        dist = supp & (np.abs(p - base) > 1.96 * se)
        v = np.where(supp, surf.value(p, base), np.nan)
        nrm = surf.make_norm()

        levels = np.linspace(-surf.rng, surf.rng, 41)
        ax.contourf(X, Y, np.clip(v, -surf.rng, surf.rng), levels=levels,
                    cmap=surf.cmap, norm=nrm, extend="both", zorder=1)
        cl = ax.contour(X, Y, v, levels=surf.lines, colors=[INK2], linewidths=0.6,
                        alpha=0.75, zorder=2)
        ax.clabel(cl, fmt=(lambda t: f"{t:+.0f}") if surf.name == "survival"
                  else (lambda t: f"{2**t:.2f}$\\times$"), fontsize=6.2,
                  inline=True, colors=INK2)
        # where the pointwise 95% band covers the base rate, veil and hatch
        undist = np.where(supp & ~dist, 1.0, 0.0)
        ax.contourf(X, Y, undist, levels=[0.5, 1.5], colors=["white"], alpha=0.66,
                    zorder=3)
        hh = ax.contourf(X, Y, undist, levels=[0.5, 1.5], colors="none",
                         hatches=["\\\\"], zorder=3)
        hh.set_edgecolor(MUTED)
        hh.set_linewidth(0.0)

        cb = fig.colorbar(ScalarMappable(norm=nrm, cmap=surf.cmap), ax=ax,
                          fraction=0.046, pad=0.02)
        cb.set_label(surf.cbar_label(base, "(smoothed)"), fontsize=7.5, color=INK2)
        surf.cbar_ticks(cb)

        _, q_sm, qr_sm = quintile_contrast(dkl, p[ix, iy])
        _, q_raw, qr_raw = quintile_contrast(dkl, f)
        share_dist = float(dist.sum() / max(supp.sum(), 1))
        pts_in_supp = float(supp[ix, iy].mean())
        vd = float(np.nanmean(np.abs(v[supp]) > surf.rng / 4))
        log(f"[SM/{surf.name}/{P['cls']}] owner-blocked CV sigma {sig:.2f} "
            f"(paired 1-SE rule {CV_SIGMAS[i1]:.2f}); design effect at that "
            f"bandwidth {deff:.2f} (gridsize {dgs} proxy lattice); "
            f"support {int(supp.sum())} of "
            f"{NB*NB} lattice cells covering {100*pts_in_supp:.1f}% of filings; "
            f"effective n median {np.median(neff[supp]):.0f}; fitted "
            f"{100*np.nanmin(p[supp]):.2f}-{100*np.nanmax(p[supp]):.2f}%; "
            f"{100*share_dist:.0f}% of the support separates from base; "
            f"{100*vd:.0f}% of it beyond a quarter-ramp; dKL Q5-Q1 {q_sm:+.2f}pp "
            f"({qr_sm:.2f}x) against {q_raw:+.2f}pp ({qr_raw:.2f}x) raw")

        style_axes(ax, f"{P['label']}  (class {P['cls']})",
                   f"{len(x):,} filings   |   base {100*base:.2f}%   |   Gaussian "
                   f"kernel, $\\sigma$={sig:.2f} KL units, owner-blocked 5-fold CV\n"
                   f"{100*share_dist:.0f}% of the drawn area separates from base "
                   f"at 95% (design effect {deff:.2f}); the rest is veiled "
                   f"and hatched")
        add_exemplars(ax, surf, P)
        diag.append(dict(surface=surf.name, cls=P["cls"], treatment="kernel_smooth",
                         n_points=len(x), base_rate=round(base, 5),
                         cells_drawn=int(supp.sum()),
                         intra_owner_correlation=round(P["rho"], 4),
                         owner_deff_at_bandwidth=round(deff, 3),
                         cv_sigma=sig, cv_sigma_1se=float(CV_SIGMAS[i1]),
                         cv_dev_min=round(float(mu[i0]), 6),
                         cv_dev_at_largest_sigma=round(float(mu[-1]), 6),
                         cv_dev_null=round(float(-(f * np.log(base)
                                                   + (1 - f) * np.log(1 - base)).mean()),
                                           6),
                         neff_median=int(np.median(neff[supp])),
                         neff_min=int(neff[supp].min()),
                         pct_points_in_support=round(100 * pts_in_supp, 1),
                         fitted_min_pct=round(100 * float(np.nanmin(p[supp])), 3),
                         fitted_max_pct=round(100 * float(np.nanmax(p[supp])), 3),
                         share_support_distinguishable=round(share_dist, 3),
                         cells_visually_distinct=int(np.nansum(
                             np.abs(v[supp]) > surf.rng / 4)),
                         share_area_visually_distinct=round(vd, 3),
                         plotted_sd_raw=round(P["raw_sd"], 3),
                         plotted_sd=round(float(np.nanstd(v[supp])), 3),
                         noise_sd_clustered=round(P["noise_sd"], 3),
                         noise_sd_binomial=round(P["noise_sd_bin"], 3),
                         dkl_q5_minus_q1_pp_raw=round(q_raw, 3),
                         dkl_q5_minus_q1_pp=round(q_sm, 3),
                         dkl_q5_over_q1_raw=round(qr_raw, 3),
                         dkl_q5_over_q1=round(qr_sm, 3)))
    plt.close(scratch)
    finish(fig, surf, HEAD[surf.name] + "\n" + SUB[surf.name]
           + ";  Nadaraya-Watson kernel regression, bandwidth by owner-blocked "
             "5-fold cross-validation,\ndrawn only where the hexbin surfaces are "
             "drawn; veiled and hatched where a pointwise 95% band covers the base",
           f"quadrant_{surf.name}_smooth")


# ================================================================ treatment 3
def figure_sig(surf, panels, diag):
    fig, axes = plt.subplots(1, 2, figsize=(15.6, 6.4))
    fig.patch.set_facecolor("white")
    cmap = surf.cmap.copy()
    cmap.set_bad(NEUTRAL)
    for ax, P in zip(axes, panels):
        x, y, f, dkl, base = P["x"], P["y"], P["f"], P["dkl"], P["base"]
        gs, mincnt = surf.hex[P["cls"]]
        off, idx, n, sel = P["off"], P["idx"], P["n"], P["sel"]
        z, pv, bh = P["z"], P["pv"], P["bh"]
        raw = np.full(len(n), np.nan)
        raw[sel] = P["ysum"][sel] / n[sel]
        v = surf.value(raw, base)
        v[~sel] = np.nan
        naive = sel & (np.abs(z) > 2)
        # every drawn cell stays in the array; the ones that fail the test are
        # masked, and the masked colour is the neutral fill, not a blank
        arr = np.where(naive, v, np.nan)
        hb = draw_cells(ax, x, y, gs, mincnt, off, sel, arr, cmap, surf.make_norm())
        hatch_cells(ax, x, y, idx, gs, naive & ~bh)

        cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label(surf.cbar_label(base, "(sig. cells only)"), fontsize=7.5,
                     color=INK2)
        surf.cbar_ticks(cb)

        K = int(sel.sum())
        exp_false = 0.0455 * K
        vq = np.where(naive, v, np.nan)
        _, q_sig, qr_sig = quintile_contrast(dkl, np.where(naive[idx], raw[idx], np.nan))
        _, q_raw, qr_raw = quintile_contrast(dkl, f)
        log(f"[SIG/{surf.name}/{P['cls']}] {K} cells; |z|>2 on "
            f"{int(naive.sum())} ({100*naive.sum()/K:.0f}%), about "
            f"{exp_false:.0f} expected by chance; BH q={FDR_Q} keeps "
            f"{int(bh.sum())}; {int((naive & ~bh).sum())} hatched; "
            f"dKL Q5-Q1 among coloured cells {q_sig:+.2f}pp ({qr_sig:.2f}x) "
            f"against {q_raw:+.2f}pp ({qr_raw:.2f}x) raw")

        nhat = int((naive & ~bh).sum())
        style_axes(ax, f"{P['label']}  (class {P['cls']})",
                   f"{len(x):,} filings   |   base {100*base:.2f}%   |   "
                   f"{int(naive.sum())} of {K} cells at $|z|>2$ "
                   f"($\\approx${exp_false:.0f} expected by chance)\n"
                   + (f"all {int(bh.sum())} survive Benjamini-Hochberg at q={FDR_Q}"
                      if nhat == 0 else
                      f"{int(bh.sum())} survive Benjamini-Hochberg at q={FDR_Q}, "
                      f"the other {nhat} are hatched")
                   + f"   |   owner design effect "
                     f"{np.median(P['deff'][sel]):.2f}")
        add_exemplars(ax, surf, P)
        diag.append(dict(surface=surf.name, cls=P["cls"], treatment="significance_mask",
                         n_points=len(x), base_rate=round(base, 5), cells_drawn=K,
                         intra_owner_correlation=round(P["rho"], 4),
                         owner_deff_median=round(float(np.median(P["deff"][sel])), 3),
                         owner_deff_max=round(float(P["deff"][sel].max()), 3),
                         cells_sig_naive=int(naive.sum()),
                         cells_sig_expected_by_chance=round(exp_false, 1),
                         cells_sig_bh=int(bh.sum()),
                         cells_hatched_naive_not_bh=int((naive & ~bh).sum()),
                         cells_neutral=int(K - naive.sum()),
                         cells_visually_distinct=int(np.nansum(
                             np.abs(vq[sel]) > surf.rng / 4)),
                         plotted_sd_raw=round(P["raw_sd"], 3),
                         plotted_sd=round(float(np.nanstd(vq[sel])), 3),
                         noise_sd_clustered=round(P["noise_sd"], 3),
                         noise_sd_binomial=round(P["noise_sd_bin"], 3),
                         dkl_q5_minus_q1_pp_raw=round(q_raw, 3),
                         dkl_q5_minus_q1_pp=round(q_sig, 3),
                         dkl_q5_over_q1_raw=round(qr_raw, 3),
                         dkl_q5_over_q1=round(qr_sig, 3)))
    finish(fig, surf, HEAD[surf.name] + "\n" + SUB[surf.name]
           + ";  cells within about two standard errors of the panel base rate "
             "drawn in neutral,\nand cells that clear $|z|>2$ but not a "
             "Benjamini-Hochberg FDR cut hatched",
           f"quadrant_{surf.name}_sig")


# ------------------------------------------------------------------- main ---
def build_frames() -> dict[str, pl.DataFrame]:
    cache = os.environ.get("QUADRANT_FRAME_CACHE")
    if cache and all((Path(cache) / f"panel_{c}.parquet").exists()
                     for c, _, _ in PANELS):
        log(f"[cache] reading panel frames from {cache}")
        return {c: pl.read_parquet(Path(cache) / f"panel_{c}.parquet")
                for c, _, _ in PANELS}
    c8 = Q.c8_first()
    wide, only8a, literal = Q.public_keys()
    frames = {}
    for cls, _, _ in PANELS:
        cov = Q.forward_coverage(cls)
        frames[cls] = Q.panel_frame(cls, c8, cov).with_columns(
            pl.col("norm").is_in(list(wide)).fill_null(False).alias("is_pub"),
            pl.col("norm").is_in(list(only8a)).fill_null(False).alias("is_8a"),
            pl.col("owner_name").is_in(list(literal)).fill_null(False).alias("is_literal"),
        )
    if cache:
        Path(cache).mkdir(parents=True, exist_ok=True)
        for cls in frames:
            frames[cls].write_parquet(Path(cache) / f"panel_{cls}.parquet")
    return frames


def prepare(surf, frames):
    """Per-panel geometry, cell counts and significance, computed once."""
    panels = []
    scratch, sax = plt.subplots()
    for cls, label, examples in PANELS:
        d = frames[cls]
        x, y, f, dkl, s = surf.sample(d)
        base = float(f.mean())
        gs, mincnt = surf.hex[cls]
        off, idx = hex_lattice(sax, x, y, gs)
        n, ysum = cell_stats(idx, f, len(off))
        assert abs(n.sum() - len(x)) < 1e-6
        sel = n >= mincnt
        _, oid = np.unique(s["norm"].fill_null("__NA__").to_numpy(),
                           return_inverse=True)
        rho = intra_owner_correlation(oid, f)
        deff, mbar = owner_design_effect(idx, oid, len(off), rho)
        neff = n / deff                       # information, not filings
        raw = np.divide(ysum, n, out=np.full(len(n), np.nan), where=n > 0)
        se = np.sqrt(base * (1 - base) / np.maximum(neff, 1))
        se_bin = np.sqrt(base * (1 - base) / np.maximum(n, 1))
        z = np.where(sel, (raw - base) / se, np.nan)
        pv = np.where(sel, 2 * NORM.sf(np.abs(np.nan_to_num(z))), np.nan)
        bh = np.zeros(len(n), bool)
        bh[sel] = bh_reject(pv[sel])
        v = surf.value(raw, base)

        def noise(sd):
            return (float(np.mean(100 * sd[sel])) if surf.name == "survival"
                    else float(np.sqrt(np.mean((sd[sel] / base) ** 2))) / np.log(2))

        noise_sd, noise_sd_bin = noise(se), noise(se_bin)
        panels.append(dict(cls=cls, label=label, _examples=examples, frame=d,
                           x=x, y=y, f=f, dkl=dkl, base=base, off=off, idx=idx,
                           oid=oid, rho=rho, n=n, neff=neff, deff=deff, mbar=mbar,
                           ysum=ysum, sel=sel, z=z, pv=pv, bh=bh,
                           raw_sd=float(np.nanstd(v[sel])),
                           noise_sd=noise_sd, noise_sd_bin=noise_sd_bin))
        log(f"[prep/{surf.name}/{cls}] {len(x):,} filings, base {100*base:.3f}%, "
            f"{int(sel.sum())} of {len(off)} cells at mincnt={mincnt}; cell n "
            f"{int(n[sel].min())}-{int(n[sel].max())} (median "
            f"{int(np.median(n[sel]))}); intra-owner correlation {rho:.3f}, cell "
            f"design effect median {np.median(deff[sel]):.2f} "
            f"(range {deff[sel].min():.2f}-{deff[sel].max():.2f}); effective cell "
            f"n {int(neff[sel].min())}-{int(neff[sel].max())} (median "
            f"{int(np.median(neff[sel]))})")
        log(f"     plotted sd {np.nanstd(v[sel]):.3f} vs noise {noise_sd:.3f} "
            f"clustered / {noise_sd_bin:.3f} binomial -> dispersion "
            f"{np.nanstd(v[sel])/noise_sd:.2f} (was "
            f"{np.nanstd(v[sel])/noise_sd_bin:.2f} against binomial)")
    plt.close(scratch)
    return panels


def main() -> int:
    frames = build_frames()
    diag: list[dict] = []
    cells_out: list[dict] = []
    for name in ("survival", "listing"):
        surf = Surface(name)
        panels = prepare(surf, frames)
        figure_eb(surf, panels, diag, cells_out)
        figure_smooth(surf, panels, diag)
        figure_sig(surf, panels, diag)
    pl.from_dicts(diag, infer_schema_length=None).write_csv(
        RES / "quadrant_surface_treatments.csv")
    pl.from_dicts(cells_out, infer_schema_length=None).write_csv(
        RES / "quadrant_surface_treatment_cells.csv")
    log("[done] quadrant_{survival,listing}_{eb,smooth,sig}.{png,pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
