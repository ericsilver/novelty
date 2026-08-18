"""Build the theme explorer: an index filtered by Nice class, and a page per theme.

Reads theme_assign_T{T}.json (what each theme dominates, by class and month, with
lifecycle outcomes) and the fitted model (what each theme is made of), and writes
a static site. Static rather than data-driven because it has to work from a local
file and from GitHub Pages with no server, and because 500 themes of monthly
series is far too much to hold in one page.

Charting decisions, per the project's data-viz rules:
  - A stacked bar per month answers "how much, split by what" over time.
  - Nice class is an identity, so the categorical palette applies, assigned in
    fixed slot order. Eight slots exist; a theme touching more classes folds the
    remainder into "Other" rather than inventing a ninth hue.
  - Colour follows the class, never its rank, so a class keeps its hue across
    every theme page.
  - Every page carries a counts table, which is what the palette's relief rule
    requires: three light-mode slots sit below 3:1 on the light surface.

Usage:  python scripts/theme_pages.py [T] [TOP_CLASSES]
Output: docs/online-appendix/themes/index.html and theme-NNNN.html
"""
from __future__ import annotations

import html
import json
import sys
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
RES = REPO / "paper" / "results"
OUT = REPO / "docs" / "online-appendix" / "themes"

T = int(sys.argv[1]) if len(sys.argv) > 1 else 500
TOPC = int(sys.argv[2]) if len(sys.argv) > 2 else 8
NWORDS = 14

LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
         "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
DARK = ["#3987e5", "#d95926", "#199e70", "#c98500",
        "#d55181", "#008300", "#9085e9", "#e66767"]
OTHER_L, OTHER_D = "#8a8f98", "#767b84"

NICE = {
    "001": "Chemicals", "002": "Paints", "003": "Cosmetics", "004": "Lubricants",
    "005": "Pharmaceuticals", "006": "Metal goods", "007": "Machines",
    "008": "Hand tools", "009": "Electrical & software", "010": "Medical apparatus",
    "011": "Heating & lighting", "012": "Vehicles", "013": "Firearms",
    "014": "Jewellery", "015": "Instruments", "016": "Paper & printed",
    "017": "Rubber & plastics", "018": "Leather", "019": "Building materials",
    "020": "Furniture", "021": "Housewares", "022": "Ropes & textiles",
    "023": "Yarns", "024": "Fabrics", "025": "Clothing", "026": "Lace & trimmings",
    "027": "Floor coverings", "028": "Toys & sport", "029": "Meat & processed foods",
    "030": "Staple foods", "031": "Agriculture", "032": "Beers & soft drinks",
    "033": "Wines & spirits", "034": "Tobacco", "035": "Advertising & retail",
    "036": "Insurance & finance", "037": "Construction & repair",
    "038": "Telecommunications", "039": "Transport", "040": "Materials treatment",
    "041": "Education", "042": "Scientific & IT", "043": "Food services",
    "044": "Medical & agriculture services", "045": "Legal & personal",
}

CSS = """
:root{--bg:#fcfcfb;--fg:#0b0b0b;--mut:#52514e;--line:#e2e2df;--warn:#c05621;--acc:#2a78d6;--card:#fff}
@media(prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#1a1a19;--fg:#fff;--mut:#c3c2b7;--line:#33332f;--warn:#e08a4f;--acc:#3987e5;--card:#232320}}
:root[data-theme=dark]{--bg:#1a1a19;--fg:#fff;--mut:#c3c2b7;--line:#33332f;--warn:#e08a4f;--acc:#3987e5;--card:#232320}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--fg);font:15px/1.55 -apple-system,Segoe UI,Roboto,sans-serif;margin:0;padding:1.6rem 2rem 4rem;max-width:82rem}
a{color:var(--acc)}
h1{font-size:1.35rem;margin:0 0 .3rem}
h2{font-size:1.02rem;margin:2rem 0 .6rem}
p.sub{color:var(--mut);margin:0 0 1.1rem;max-width:58rem}
.controls{display:flex;gap:.8rem;flex-wrap:wrap;align-items:center;margin:1rem 0}
select,input{padding:.45rem .6rem;border:1px solid var(--line);border-radius:6px;background:var(--card);color:var(--fg);font:inherit}
input{min-width:20rem}
.stats{display:flex;gap:2rem;flex-wrap:wrap;margin:.4rem 0 1rem}
.stat b{display:block;font-size:1.45rem;font-variant-numeric:tabular-nums}
.stat span{color:var(--mut);font-size:.82rem}
.wrap{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:14px}
th,td{text-align:left;padding:.42rem .6rem;border-bottom:1px solid var(--line);vertical-align:top}
th{position:sticky;top:0;background:var(--bg);color:var(--mut);font-weight:600;font-size:.78rem;text-transform:uppercase;letter-spacing:.04em}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.legend{display:flex;gap:1rem;flex-wrap:wrap;margin:.5rem 0 .2rem;font-size:13px;color:var(--mut)}
.legend i{display:inline-block;width:11px;height:11px;border-radius:2px;margin-right:.35rem;vertical-align:-1px}
figure{margin:0}
figcaption{color:var(--mut);font-size:13px;margin-top:.5rem;max-width:58rem}
.back{font-size:14px;margin-bottom:.8rem;display:block}
rect:hover{opacity:.72}
"""

JS = """
var tb=document.getElementById('tb');
var rows=Array.prototype.slice.call(tb.querySelectorAll('tr'));
var cls=document.getElementById('cls'), q=document.getElementById('q');
var cnt=document.getElementById('cnt');
function countFor(r){
  if(!cls.value) return 0;
  var m=JSON.parse(r.getAttribute('data-classes'));
  return m[cls.value]||0;
}
function apply(){
  var v=q.value.toLowerCase().trim(), shown=0;
  var vis=rows.filter(function(r){
    var okw=!v||r.getAttribute('data-words').indexOf(v)>-1;
    var okc=!cls.value||countFor(r)>0;
    return okw&&okc;
  });
  rows.forEach(function(r){r.style.display='none';});
  if(cls.value) vis.sort(function(a,b){return countFor(b)-countFor(a);});
  vis.forEach(function(r){r.style.display='';tb.appendChild(r);shown++;});
  cnt.textContent=shown+' of '+rows.length+' themes';
}
cls.addEventListener('change',apply);
q.addEventListener('input',apply);
apply();
"""


def cname(c: str) -> str:
    return (c + " " + NICE.get(c, "")).strip()


def esc(s) -> str:
    return html.escape(str(s))


def slot_vars(n: int) -> str:
    """Fixed-slot colour assignment; the ninth series onward is Other."""
    rl = ";".join("--s%d:%s" % (i, LIGHT[i] if i < len(LIGHT) else OTHER_L)
                  for i in range(n))
    rd = ";".join("--s%d:%s" % (i, DARK[i] if i < len(DARK) else OTHER_D)
                  for i in range(n))
    return (":root{" + rl + "}"
            "@media(prefers-color-scheme:dark){:root:not([data-theme=light]){"
            + rd + "}}"
            ":root[data-theme=dark]{" + rd + "}")


def stacked_svg(series, cols, months, W=900, H=270):
    if not months:
        return "<p>No filings in range.</p>"
    pad_l, pad_b, pad_t = 48, 34, 8
    iw = W - pad_l - 10
    ih = H - pad_b - pad_t
    tot = [sum(series[c].get(mo, 0) for c in cols) for mo in months]
    mx = max(tot) or 1
    bw = iw / len(months)
    gap = 0.35 if bw > 3 else 0.0
    o = ['<svg viewBox="0 0 %d %d" width="100%%" role="img" '
         'aria-label="Filings per month by Nice class">' % (W, H)]
    for gy in range(5):
        y = pad_t + ih - ih * gy / 4.0
        o.append('<line x1="%d" x2="%d" y1="%.1f" y2="%.1f" stroke="var(--line)" '
                 'stroke-width="1"/>' % (pad_l, W - 10, y, y))
        o.append('<text x="%d" y="%.1f" text-anchor="end" font-size="11" '
                 'fill="var(--mut)">%s</text>'
                 % (pad_l - 6, y + 4, format(int(mx * gy / 4.0), ",")))
    for i, mo in enumerate(months):
        x = pad_l + i * bw
        acc = 0.0
        for ci, c in enumerate(cols):
            v = series[c].get(mo, 0)
            if not v:
                continue
            h = ih * v / mx
            y = pad_t + ih - acc - h
            label = cname(c) if c != "Other" else "Other classes"
            o.append('<rect x="%.2f" y="%.2f" width="%.2f" height="%.2f" '
                     'fill="var(--s%d)" rx="1"><title>%s-%s &middot; %s &middot; %s</title></rect>'
                     % (x + gap, y, max(bw - 2 * gap, 0.6), max(h - 1, 0.5), ci,
                        esc(mo[:4]), esc(mo[4:]), esc(label), format(v, ",")))
            acc += h
    step = max(1, len(months) // 8)
    for i in range(0, len(months), step):
        o.append('<text x="%.1f" y="%d" text-anchor="middle" font-size="11" '
                 'fill="var(--mut)">%s</text>'
                 % (pad_l + i * bw + bw / 2, H - 12, esc(months[i][:4])))
    o.append("</svg>")
    return "".join(o)


def shell(title, body, extra_css=""):
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>' + esc(title) + '</title><style>' + CSS + extra_css
            + '</style></head><body>' + body + '</body></html>')


def main() -> int:
    src = RES / ("theme_assign_T%d.json" % T)
    if not src.exists():
        print("missing %s; run scripts/theme_assign.py %d first" % (src, T),
              file=sys.stderr)
        return 1
    D = json.loads(src.read_text())
    outc, mon, caps = D["outcomes"], D["months"], D["classes"]

    mp = PROC / ("topic_model.joblib" if T == 50 else "topic_model_T%d.joblib" % T)
    m = joblib.load(mp)
    inv = {v: k for k, v in m["vocabulary"].items()}
    comp = m["lda"].components_
    top = [", ".join(inv[j] for j in np.argsort(comp[k])[::-1][:NWORDS])
           for k in range(T)]
    del m, comp

    OUT.mkdir(parents=True, exist_ok=True)
    all_classes = sorted({c for t in outc.values() for c in t["by_class"]})

    for k in range(T):
        sk = str(k)
        o = outc.get(sk, {"n": 0, "reg": 0, "sec": 0, "failed": 0, "by_class": {}})
        byc = o["by_class"]
        ranked = sorted(byc.items(), key=lambda kv: -kv[1])
        keep = [c for c, _ in ranked[:TOPC]]
        rest = [c for c, _ in ranked[TOPC:]]
        msrc = mon.get(sk, {})
        series = {c: dict(msrc.get(c, {})) for c in keep}
        if rest:
            agg = defaultdict(int)
            for c in rest:
                for mo, v in msrc.get(c, {}).items():
                    agg[mo] += v
            series["Other"] = dict(agg)
        cols = keep + (["Other"] if rest else [])
        months = sorted({mo for c in cols for mo in series[c]})
        months = [mo for mo in months if "199001" <= mo <= "202512"]

        leg = "".join(
            '<span><i style="background:var(--s%d)"></i>%s</span>'
            % (i, esc(cname(c) if c != "Other" else
                      "Other (%d classes)" % len(rest)))
            for i, c in enumerate(cols))
        rows = "".join(
            '<tr><td>%s</td><td class="n">%s</td></tr>'
            % (esc(cname(c)), format(v, ",")) for c, v in ranked)
        n = o["n"] or 1
        reg = max(o["reg"], 1)
        body = (
            '<a class="back" href="index.html">&larr; all themes</a>'
            '<h1>Theme %d</h1><p class="sub">%s</p>'
            '<div class="stats">'
            '<div class="stat"><b>%s</b><span>filings dominated</span></div>'
            '<div class="stat"><b>%.1f%%</b><span>reached registration</span></div>'
            '<div class="stat"><b>%.1f%%</b><span>failed first gate, of registered</span></div>'
            '<div class="stat"><b>%.2f%%</b><span>owner SEC-reporting</span></div>'
            '<div class="stat"><b>%d</b><span>classes present</span></div></div>'
            '<h2>Filings per month, by Nice class</h2>'
            '<div class="legend">%s</div>'
            '<figure>%s<figcaption>Filings for which this is the highest-weighted '
            'theme, by filing month. Hover a segment for its class and count. '
            'Classes beyond the top %d are pooled as Other.</figcaption></figure>'
            '<h2>All classes</h2><div class="wrap"><table><thead><tr>'
            '<th>Nice class</th><th class="n">Filings dominated</th></tr></thead>'
            '<tbody>%s</tbody></table></div>'
            % (k, esc(top[k]), format(o["n"], ","), 100.0 * o["reg"] / n,
               100.0 * o["failed"] / reg, 100.0 * o["sec"] / n, len(byc),
               leg, stacked_svg(series, cols, months), TOPC, rows))
        (OUT / ("theme-%04d.html" % k)).write_text(
            shell("Theme %d" % k, body, slot_vars(len(cols))), encoding="utf-8")

    opts = "".join('<option value="%s">%s</option>' % (c, esc(cname(c)))
                   for c in all_classes)
    trs = []
    for k in range(T):
        o = outc.get(str(k))
        if not o:
            continue
        n = o["n"] or 1
        reg = max(o["reg"], 1)
        trs.append(
            '<tr data-classes=\'%s\' data-words="%s">'
            '<td class="n"><a href="theme-%04d.html">%d</a></td>'
            '<td>%s</td><td class="n">%s</td><td class="n">%.1f%%</td>'
            '<td class="n">%.1f%%</td><td class="n">%.2f%%</td></tr>'
            % (json.dumps(o["by_class"]), esc(top[k].lower()), k, k,
               esc(top[k]), format(o["n"], ","), 100.0 * o["reg"] / n,
               100.0 * o["failed"] / reg, 100.0 * o["sec"] / n))
    total = sum(v["used"] for v in caps.values())
    sampled = [c for c, v in caps.items() if v["used"] < v["total"]]
    body = (
        '<h1>Themes at T&nbsp;=&nbsp;%d</h1>'
        '<p class="sub">Every theme in the fitted model, with the words that lead '
        'it and what became of the filings it dominates. Each filing is counted '
        'once, under its single highest-weighted theme. Choose a Nice class to '
        'keep only the themes present in it, ordered by how many of its filings '
        'they dominate. Click a theme for its monthly trend split by class.</p>'
        '<div class="controls">'
        '<label>Nice class <select id="cls"><option value="">All classes</option>'
        '%s</select></label>'
        '<input id="q" placeholder="filter by word, e.g. blockchain, kombucha">'
        '<span id="cnt" style="color:var(--mut);font-size:13px"></span></div>'
        '<div class="wrap"><table><thead><tr>'
        '<th class="n">#</th><th>Leading words</th><th class="n">Filings</th>'
        '<th class="n">Registered</th><th class="n">Gate failure</th>'
        '<th class="n">SEC-reporting</th></tr></thead>'
        '<tbody id="tb">%s</tbody></table></div>'
        '<p class="sub" style="margin-top:1.4rem;font-size:13px">%s filings '
        'assigned. %d of %d classes were sampled to %s filings; the rest are '
        'complete. Gate failure is a dated non-use cancellation at registration '
        'age 4.0&ndash;8.5 years, as a share of registrations.</p>'
        '<script>%s</script>'
        % (T, opts, "".join(trs), format(total, ","), len(sampled), len(caps),
           format(D["cap"], ","), JS))
    (OUT / "index.html").write_text(shell("Themes at T=%d" % T, body),
                                    encoding="utf-8")
    print("[pages] %d theme pages + index -> %s" % (T, OUT), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
