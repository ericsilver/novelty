"""Convert paper/results/big_firm_kl_v2.csv into a tabular-only .tex
fragment for inclusion under \\input{} in the main paper. The .tex file
emitted by the upstream big_firm_kl_v2.py wraps the table in a full
\\begin{table}...\\end{table} environment, which collides with the
\\begin{table} wrapper already in main.tex.
"""

from __future__ import annotations

from pathlib import Path
import polars as pl

REPO = Path(__file__).resolve().parents[1]
OUT  = REPO / "paper" / "results"


def main() -> int:
    d = pl.read_csv(OUT / "big_firm_kl_v2.csv").filter(pl.col("n_clean_filings") > 0)
    lines = [
        r"\begin{tabular}{lllrrrrrrr}",
        r"\toprule",
        r"\textbf{cohort} & \textbf{brand} & \textbf{founded} & \textbf{n} & "
        r"\textbf{mn\_pros} & \textbf{mn\_retr} & \textbf{mn\_$\Delta$KL} & "
        r"\textbf{1st\_pros} & \textbf{1st\_retr} & \textbf{1st\_$\Delta$KL} \\",
        r"\midrule",
    ]
    for r in d.iter_rows(named=True):
        c = r"\textsc{win}" if r["cohort"] == "winner" else r"\textsc{fail}"
        brand = r["brand"].replace("&", r"\&").replace("_", r"\_")
        lines.append(
            f"{c} & {brand} & {r['founded']} & {r['n_clean_filings']:,} & "
            f"{r['mean_pros']:.2f} & {r['mean_retr']:.2f} & {r['mean_dkl']:+.3f} & "
            f"{r['first_real_pros']:.2f} & {r['first_real_retr']:.2f} & "
            f"{r['first_real_dkl']:+.2f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "big_firm_kl_v2.tex").write_text("\n".join(lines))
    print(f"wrote {OUT / 'big_firm_kl_v2.tex'}  ({d.height} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
