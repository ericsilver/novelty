"""Build the theme explorer: an index filtered by Nice class, and a page per theme.

Reads theme_assign_T{T}.json (what each theme dominates, by class and month, with
lifecycle outcomes) and the fitted model (what each theme is made of), and writes
a static site. Static rather than data-driven because it has to work from a local
file and from GitHub Pages with no server, and because 500 themes of monthly
series is far too much to hold in one page.

Charting decisions, per the project's data-viz rules:
  - One line per Nice class over quarters, plus a total. Lines rather than a
    stacked bar because the question is "when was this used, and by whom",
    which compares trajectories, and stacking makes every series but the
    bottom one unreadable. Quarters rather than months because a theme with a
    handful of filings a month is noise at monthly resolution.
  - The total is drawn in ink at low opacity, not given a categorical hue: it
    is an aggregate of the other series, not another category.
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
.help{display:inline-flex;align-items:center;justify-content:center;width:18px;height:18px;border-radius:50%;border:1px solid var(--line);color:var(--mut);font-size:12px;cursor:help;position:relative}
.help:hover,.help:focus{color:var(--fg);border-color:var(--mut);outline:none}
.help .tip{display:none;position:absolute;left:50%;transform:translateX(-50%);top:24px;width:30rem;max-width:70vw;background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:8px;padding:.7rem .85rem;font-size:13px;line-height:1.45;z-index:20;box-shadow:0 6px 24px rgba(0,0,0,.18);text-align:left}
.help:hover .tip,.help:focus .tip{display:block}
th.sort{cursor:pointer;user-select:none}
th.sort:hover{color:var(--fg)}
th.sort i{font-style:normal;opacity:.35;margin-left:.3rem}
th.sort i::after{content:"↕"}
th.sort[data-dir="asc"] i{opacity:1}
th.sort[data-dir="asc"] i::after{content:"↑"}
th.sort[data-dir="desc"] i{opacity:1}
th.sort[data-dir="desc"] i::after{content:"↓"}
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
var topn=document.getElementById('topn');
var cnt=document.getElementById('cnt');
function countFor(r){
  if(!cls.value) return 0;
  var m=JSON.parse(r.getAttribute('data-classes'));
  return m[cls.value]||0;
}
var sortKey=null, sortDir='desc';
function num(r,k){
  var c=r.children[k];
  var t=(c?c.textContent:'').replace(/[,%]/g,'');
  var f=parseFloat(t);
  return isNaN(f)?-Infinity:f;
}
function apply(){
  var v=q.value.toLowerCase().trim(), shown=0;
  var vis=rows.filter(function(r){
    var okw=!v||r.getAttribute('data-words').indexOf(v)>-1;
    var okc=!cls.value||countFor(r)>0;
    var n=+topn.value;
    var okn=!n||(+r.getAttribute('data-rank'))<n;
    return okw&&okc&&okn;
  });
  rows.forEach(function(r){r.style.display='none';});
  if(sortKey!==null){
    var sgn=(sortDir==='asc')?1:-1;
    vis.sort(function(a,b){return sgn*(num(a,sortKey)-num(b,sortKey));});
  } else if(cls.value){
    // no explicit sort: with a class chosen, order by that class's filings
    vis.sort(function(a,b){return countFor(b)-countFor(a);});
  }
  vis.forEach(function(r){r.style.display='';tb.appendChild(r);shown++;});
  cnt.textContent=shown+' of '+rows.length+' themes shown';
}
Array.prototype.forEach.call(document.querySelectorAll('th.sort'),function(th){
  th.addEventListener('click',function(){
    var k=+th.getAttribute('data-k');
    if(sortKey===k){ sortDir=(sortDir==='desc')?'asc':'desc'; }
    else { sortKey=k; sortDir='desc'; }
    Array.prototype.forEach.call(document.querySelectorAll('th.sort'),
      function(o){o.removeAttribute('data-dir');});
    th.setAttribute('data-dir',sortDir);
    apply();
  });
});
cls.addEventListener('change',apply);
topn.addEventListener('change',apply);
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


def to_quarters(series, cols):
    """Monthly counts to quarterly. Monthly bars are unreadable for a theme with
    a few filings a month; quarters give each point enough mass to mean
    something without hiding a trend."""
    out = {}
    for c in cols:
        q = {}
        for mo, v in series[c].items():
            key = "%sQ%d" % (mo[:4], (int(mo[4:6]) - 1) // 3 + 1)
            q[key] = q.get(key, 0) + v
        out[c] = q
    return out


def line_svg(series, cols, quarters, W=900, H=300):
    """One line per Nice class plus a total.

    A line rather than a stacked bar because the question a reader brings to a
    theme page is "when was this used, and by whom" -- which is a comparison of
    trajectories, and stacking makes every series except the bottom one impossible
    to read off. The total is drawn in ink rather than given a categorical hue,
    because it is an aggregate of the others and not another category.
    """
    if not quarters:
        return "<p>No filings in range.</p>"
    pad_l, pad_b, pad_t, pad_r = 52, 34, 10, 12
    iw = W - pad_l - pad_r
    ih = H - pad_b - pad_t
    tot = [sum(series[c].get(q, 0) for c in cols) for q in quarters]
    mx = max(tot) or 1
    n = len(quarters)
    x = lambda i: pad_l + (iw * i / max(n - 1, 1))
    yv = lambda v: pad_t + ih - ih * v / mx
    o = ['<svg viewBox="0 0 %d %d" width="100%%" role="img" '
         'aria-label="Filings per quarter by Nice class">' % (W, H)]
    for gy in range(5):
        yy = pad_t + ih - ih * gy / 4.0
        o.append('<line x1="%d" x2="%d" y1="%.1f" y2="%.1f" stroke="var(--line)" '
                 'stroke-width="1"/>' % (pad_l, W - pad_r, yy, yy))
        o.append('<text x="%d" y="%.1f" text-anchor="end" font-size="11" '
                 'fill="var(--mut)">%s</text>'
                 % (pad_l - 6, yy + 4, format(int(mx * gy / 4.0), ",")))
    # total first, so class lines sit above it
    pts = " ".join("%.1f,%.1f" % (x(i), yv(tot[i])) for i in range(n))
    o.append('<polyline points="%s" fill="none" stroke="var(--fg)" '
             'stroke-width="2.5" stroke-opacity="0.35" stroke-linejoin="round"/>' % pts)
    for ci, c in enumerate(cols):
        pts = " ".join("%.1f,%.1f" % (x(i), yv(series[c].get(q, 0)))
                       for i, q in enumerate(quarters))
        o.append('<polyline points="%s" fill="none" stroke="var(--s%d)" '
                 'stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>'
                 % (pts, ci))
    # one hover column per quarter, wider than the marks, listing every series
    colw = iw / max(n - 1, 1)
    for i, q in enumerate(quarters):
        rows = "&#10;".join(
            "%s: %s" % (cname(c) if c != "Other" else "Other classes",
                        format(series[c].get(q, 0), ","))
            for c in cols if series[c].get(q, 0))
        o.append('<rect x="%.1f" y="%d" width="%.1f" height="%d" fill="transparent">'
                 '<title>%s&#10;%s&#10;Total: %s</title></rect>'
                 % (x(i) - colw / 2, pad_t, max(colw, 2), ih, esc(q), rows,
                    format(tot[i], ",")))
    step = max(1, n // 8)
    for i in range(0, n, step):
        o.append('<text x="%.1f" y="%d" text-anchor="middle" font-size="11" '
                 'fill="var(--mut)">%s</text>' % (x(i), H - 12, esc(quarters[i][:4])))
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
        series = {c: {mo: v for mo, v in series[c].items()
                      if "199001" <= mo <= "202512"} for c in cols}
        series = to_quarters(series, cols)
        quarters = sorted({q for c in cols for q in series[c]})

        leg = "".join(
            '<span><i style="background:var(--s%d)"></i>%s</span>'
            % (i, esc(cname(c) if c != "Other" else
                      "Other (%d classes)" % len(rest)))
            for i, c in enumerate(cols)) + (
            '<span><i style="background:var(--fg);opacity:.35"></i>Total</span>')
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
            '<h2>Filings per quarter, by Nice class</h2>'
            '<div class="legend">%s</div>'
            '<figure>%s<figcaption>Filings for which this is the highest-weighted '
            'theme, by filing quarter. Hover anywhere in a quarter for every '
            'class and the total. Classes beyond the top %d are pooled as '
            'Other.</figcaption></figure>'
            '<h2>All classes</h2><div class="wrap"><table><thead><tr>'
            '<th>Nice class</th><th class="n">Filings dominated</th></tr></thead>'
            '<tbody>%s</tbody></table></div>'
            % (k, esc(top[k]), format(o["n"], ","), 100.0 * o["reg"] / n,
               100.0 * o["failed"] / reg, 100.0 * o["sec"] / n, len(byc),
               leg, line_svg(series, cols, quarters), TOPC, rows))
        (OUT / ("theme-%04d.html" % k)).write_text(
            shell("Theme %d" % k, body, slot_vars(len(cols))), encoding="utf-8")

    opts = "".join('<option value="%s">%s</option>' % (c, esc(cname(c)))
                   for c in all_classes)
    # rank by filings dominated, so "the 100 most-used themes" is well defined
    order = sorted(range(T), key=lambda k: -(outc.get(str(k), {}).get("n", 0)))
    rank = {k: i for i, k in enumerate(order)}
    trs = []
    for k in range(T):
        o = outc.get(str(k))
        if not o:
            continue
        n = o["n"] or 1
        reg = max(o["reg"], 1)
        trs.append(
            '<tr data-classes=\'%s\' data-words="%s" data-rank="%d">'
            '<td class="n"><a href="theme-%04d.html">%d</a></td>'
            '<td>%s</td><td class="n">%s</td><td class="n">%.1f%%</td>'
            '<td class="n">%.1f%%</td><td class="n">%.2f%%</td></tr>'
            % (json.dumps(o["by_class"]), esc(top[k].lower()), rank[k], k, k,
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
        'they dominate, or click any numeric column to sort by it. Click a theme '
        'for its quarterly trend split by class.</p>'
        '<div class="controls">'
        '<label>Nice class <select id="cls"><option value="">All classes</option>'
        '%s</select></label>'
        '<label>Show <select id="topn">'
        '<option value="100" selected>100 most-used themes</option>'
        '<option value="250">250 most-used</option>'
        '<option value="0">all %d</option>'
        '</select></label>'
        '<input id="q" placeholder="filter by word, e.g. blockchain, insurance">'
        '<span class="help" tabindex="0">?<span class="tip">Every filing is '
        'matched to one of %d themes, but rates computed on a theme that '
        'dominates only a handful of filings are noise: one owner reaching SEC '
        'reporting can move that column by tens of percentage points. The list '
        'is therefore limited to the most-used themes by default, so the '
        'registration, gate-failure and SEC-reporting columns are read on '
        'themes with enough filings to mean something. Widen it and the tail '
        'will sort to the top of any percentage column for that reason alone.'
        '</span></span>'
        '<span id="cnt" style="color:var(--mut);font-size:13px"></span></div>'
        '<div class="wrap"><table><thead><tr>'
        '<th class="n sort" data-k="0">#<i></i></th>'
        '<th>Leading words</th>'
        '<th class="n sort" data-k="2">Filings<i></i></th>'
        '<th class="n sort" data-k="3">Registered<i></i></th>'
        '<th class="n sort" data-k="4">Gate failure<i></i></th>'
        '<th class="n sort" data-k="5">SEC-reporting<i></i></th></tr></thead>'
        '<tbody id="tb">%s</tbody></table></div>'
        '<p class="sub" style="margin-top:1.4rem;font-size:13px">%s filings '
        'assigned. %d of %d classes were sampled to %s filings; the rest are '
        'complete. Gate failure is a dated non-use cancellation at registration '
        'age 4.0&ndash;8.5 years, as a share of registrations.</p>'
        '<script>%s</script>'
        % (T, opts, T, T, "".join(trs), format(total, ","), len(sampled),
           len(caps), format(D["cap"], ","), JS))
    (OUT / "index.html").write_text(shell("Themes at T=%d" % T, body),
                                    encoding="utf-8")
    print("[pages] %d theme pages + index -> %s" % (T, OUT), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
