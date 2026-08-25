"""Read-only check: replicate wave_timing.py joiner counts. Prints j.height (all joiners across all episodes) and count within >=300 episodes. Writes nothing."""
import json, sys
from pathlib import Path
import polars as pl

REPO = Path(__file__).resolve().parents[2]
PROC = REPO / "data" / "processed"
RES = REPO / "paper" / "results"
CACHE = RES / "theme_novelty_cache"
CLASSES = [f"{i:03d}" for i in range(1, 46)]
LEVEL, DOUBLE, PRIOR_YEARS, ENTRY_YEARS = 0.01, 2.0, 3, 6
REG_LO, REG_HI = 2002, 2018
MIN_N = 300

all_ = pl.concat([pl.read_parquet(CACHE / f"{c}.parquet") for c in CLASSES if (CACHE / f"{c}.parquet").exists()])
thin = all_.filter(pl.col("thin") & pl.col("fy").is_between(1986, 2024))
tot = thin.group_by(["cls", "fy"]).len().rename({"len": "N"})
sh = thin.group_by(["theme", "cls", "fy"]).len().join(tot, on=["cls", "fy"]).with_columns(
    (pl.col("len") / pl.col("N")).alias("share")).sort(["theme", "cls", "fy"])
grid = sh.select("theme", "cls").unique().join(pl.DataFrame({"fy": list(range(1986, 2025))}), how="cross")
shg = grid.join(sh, on=["theme", "cls", "fy"], how="left").with_columns(
    pl.col("share").fill_null(0.0)).sort(["theme", "cls", "fy"])
shg = shg.with_columns(pl.col("share").rolling_mean(window_size=PRIOR_YEARS).shift(1).over(["theme", "cls"]).alias("prior"))
surge = shg.filter((pl.col("share") >= LEVEL) & (pl.col("prior") > 0)
                   & (pl.col("share") >= DOUBLE * pl.col("prior"))).group_by(["theme", "cls"]).agg(
    pl.col("fy").min().alias("surge_y"))
print(f"episodes: {surge.height}")

g = all_.filter(pl.col("ry").is_between(REG_LO, REG_HI))
j = g.join(surge, on=["theme", "cls"], how="inner").with_columns(
    (pl.col("fy") - pl.col("surge_y")).alias("e")).filter(pl.col("e").is_between(0, ENTRY_YEARS - 1))
print(f"all joiners (j.height): {j.height:,}")

sizes = j.group_by(["theme", "cls"]).len()
big = sizes.filter(pl.col("len") >= MIN_N)
print(f"episodes >= {MIN_N}: {big.height}")
print(f"joiners in those episodes: {big['len'].sum():,}")
