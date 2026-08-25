"""Rebuild the pooled all-themes row of gate_era_tech_themes.tex from the JSON.

Idempotent: strips any prior 'All themes' row before inserting.
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "paper" / "results"
ERAS = ["1995-1999", "2000-2004", "2005-2007", "2008-2014"]


def fmt(r):
    return f"${100*r['lift']:+.2f}$ ({100*r['se']:.2f})" if r else "---"


j = json.loads((RES / "gate_era_tech_themes.json").read_text())
lines = [l for l in (RES / "gate_era_tech_themes.tex").read_text().splitlines()
         if "All themes" not in l]
# drop a dangling midrule immediately before bottomrule left by a prior patch
i = lines.index(r"\bottomrule")
if lines[i - 1] == r"\midrule":
    del lines[i - 1]
    i -= 1
pooled_n = sum(j["pooled_era"][lab]["raw"]["n"] for lab in ERAS)
times = "$" + "\\times" + "$"
row = (f"All themes, within theme{times}class{times}cohort & {pooled_n:,} & "
       + " & ".join(fmt(j["pooled_era"][lab]["within_theme_class_cohort"]) for lab in ERAS)
       + " \\\\")
lines[i:i] = [r"\midrule", row]
(RES / "gate_era_tech_themes.tex").write_text("\n".join(lines) + "\n")
print(row)
