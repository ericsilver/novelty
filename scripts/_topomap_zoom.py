"""Shared helpers for the zoom + contour-overlay variants of the
survival- and profit-topomaps.

Two rendering modes:

* ``zoom_panel`` — same per-cell heatmap as the parent, but with axes
  cropped to the lower-left commonplace corner (default p5–p55 of each
  KL axis) and the colormap normalised to the central percentile of
  cell values rather than the per-cell extremes.
* ``contour_panel`` — Gaussian-smoothed version of the heatmap with
  isoscore lines overlaid; sparse cells are interpolated from
  neighbours so the diagonal "valley" reads continuously.
"""

from __future__ import annotations

import numpy as np
from matplotlib.colors import Normalize, LogNorm
from scipy import ndimage


def _percentile_norm(H: np.ndarray, lo_p: float = 5, hi_p: float = 95,
                     centre: float | None = None):
    vals = H[~np.isnan(H)]
    if vals.size == 0:
        return Normalize(vmin=0, vmax=1)
    lo = np.percentile(vals, lo_p)
    hi = np.percentile(vals, hi_p)
    if centre is not None:
        absmax = max(abs(lo - centre), abs(hi - centre))
        return Normalize(vmin=centre - absmax, vmax=centre + absmax)
    if lo == hi:
        hi = lo + 1e-9
    return Normalize(vmin=lo, vmax=hi)


def smooth_for_contour(H: np.ndarray, sigma_cells: float = 1.5,
                       min_support_cells: int = 1) -> np.ndarray:
    """Fill NaN cells from neighbours, gaussian-smooth, then re-mask cells
    that had no support within a radius of `min_support_cells`.

    Returns float64 array, NaN where no smoothed value should be drawn.
    """
    mask = ~np.isnan(H)
    if mask.sum() == 0:
        return np.full_like(H, np.nan)
    # Fill NaN with the nearest-neighbour value before smoothing
    indices = ndimage.distance_transform_edt(~mask, return_distances=False,
                                              return_indices=True)
    filled = H[tuple(indices)]
    smoothed = ndimage.gaussian_filter(filled, sigma=sigma_cells, mode="nearest")
    # Mask cells that had no real support nearby
    support_dist = ndimage.distance_transform_edt(~mask)
    smoothed = np.where(support_dist <= min_support_cells * 4, smoothed, np.nan)
    return smoothed


def crop_to_lower_left(p_edges: np.ndarray, r_edges: np.ndarray,
                       H: np.ndarray, p_max_pct: float = 0.55,
                       r_max_pct: float = 0.55,
                       p_min_pct: float = 0.0, r_min_pct: float = 0.0):
    """Crop the heatmap and edges to the lower-left p_min_pct..p_max_pct
    of each axis.  Returns the cropped (p_edges, r_edges, H_crop).
    """
    n_p = len(p_edges) - 1
    n_r = len(r_edges) - 1
    p_lo = int(np.floor(p_min_pct * n_p))
    p_hi = max(p_lo + 2, int(np.ceil(p_max_pct * n_p)))
    r_lo = int(np.floor(r_min_pct * n_r))
    r_hi = max(r_lo + 2, int(np.ceil(r_max_pct * n_r)))
    return (p_edges[p_lo:p_hi + 1], r_edges[r_lo:r_hi + 1],
            H[r_lo:r_hi, p_lo:p_hi])


def draw_zoom_panel(ax, p_edges, r_edges, H, *, cmap, title,
                    diverging_centre: float | None = None,
                    log_scale: bool = False,
                    n_contour_levels: int = 8,
                    p_max_pct: float = 0.55, r_max_pct: float = 0.55):
    """Draw a zoomed-in pcolormesh of the lower-left quadrant with
    percentile-clipped colour limits and isoscore contour lines."""
    p_e, r_e, H_c = crop_to_lower_left(p_edges, r_edges, H,
                                        p_max_pct=p_max_pct,
                                        r_max_pct=r_max_pct)
    if log_scale:
        vmax = np.nanmax(H_c) if np.any(~np.isnan(H_c)) else 1
        norm = LogNorm(vmin=1, vmax=max(vmax, 2))
    else:
        norm = _percentile_norm(H_c, lo_p=5, hi_p=95, centre=diverging_centre)

    im = ax.pcolormesh(p_e, r_e, H_c, cmap=cmap, shading="flat", norm=norm)
    # Contour overlay on smoothed surface
    smooth = smooth_for_contour(H_c, sigma_cells=1.2)
    if np.any(~np.isnan(smooth)):
        p_c = 0.5 * (p_e[:-1] + p_e[1:])
        r_c = 0.5 * (r_e[:-1] + r_e[1:])
        try:
            cs = ax.contour(p_c, r_c, smooth, levels=n_contour_levels,
                            colors="white", linewidths=0.5, alpha=0.65)
            ax.clabel(cs, inline=True, fontsize=6, fmt="%.2f", inline_spacing=2)
        except (ValueError, IndexError):
            pass

    diag_lo = max(p_e[0], r_e[0])
    diag_hi = min(p_e[-1], r_e[-1])
    ax.plot([diag_lo, diag_hi], [diag_lo, diag_hi], color="black",
            linewidth=0.9, linestyle="--", alpha=0.85)
    ax.set_xlim(p_e[0], p_e[-1])
    ax.set_ylim(r_e[0], r_e[-1])
    ax.set_xlabel(r"Mean prospective KL  (firm)")
    ax.set_ylabel(r"Mean retrospective KL  (firm)")
    ax.set_title(title, fontsize=10)
    return im


def draw_contour_panel(ax, p_edges, r_edges, H, *, cmap, title,
                       diverging_centre: float | None = None,
                       sigma_cells: float = 1.5,
                       n_contour_levels: int = 12):
    """Full-range smoothed contour-fill plot with isoscore lines.
    No zoom — meant to read like a topographic map."""
    smooth = smooth_for_contour(H, sigma_cells=sigma_cells)
    p_c = 0.5 * (p_edges[:-1] + p_edges[1:])
    r_c = 0.5 * (r_edges[:-1] + r_edges[1:])
    norm = _percentile_norm(H, lo_p=5, hi_p=95, centre=diverging_centre)
    masked = np.ma.masked_invalid(smooth)
    cs_fill = ax.contourf(p_c, r_c, masked, levels=n_contour_levels,
                          cmap=cmap, norm=norm, extend="both")
    if np.any(~np.isnan(smooth)):
        cs = ax.contour(p_c, r_c, smooth, levels=n_contour_levels,
                        colors="black", linewidths=0.4, alpha=0.5)
        ax.clabel(cs, inline=True, fontsize=6, fmt="%.2f", inline_spacing=2)
    diag_lo = max(p_edges[0], r_edges[0])
    diag_hi = min(p_edges[-1], r_edges[-1])
    ax.plot([diag_lo, diag_hi], [diag_lo, diag_hi], color="black",
            linewidth=0.8, linestyle="--", alpha=0.85)
    ax.set_xlim(p_edges[0], p_edges[-1])
    ax.set_ylim(r_edges[0], r_edges[-1])
    ax.set_xlabel(r"Mean prospective KL  (firm)")
    ax.set_ylabel(r"Mean retrospective KL  (firm)")
    ax.set_title(title, fontsize=10)
    return cs_fill
