"""Render the paper as speakable prose, and synthesise it to audio.

A LaTeX paper read aloud by a screen reader is unlistenable: it says "backslash
emph", spells out every citation key, reads tables cell by cell, and pronounces
"$t = 8.3$" as a string of symbols. This strips the document to the prose a
person would actually want narrated, rewrites the notation into words, and hands
the result to the Windows speech engine.

What is dropped: tables, figures, captions, the bibliography, and the equations,
none of which survive narration. What is rewritten: percentages, per-sigma
units, KL symbols, en-dash ranges, citation commands, and the abbreviations that
read badly aloud.

Sections can be selected so a listener can take the argument without the
appendices, which are roughly two thirds of the running time.

Usage:
  python scripts/paper_to_audio.py                 # body only (no appendices)
  python scripts/paper_to_audio.py --all           # everything
  python scripts/paper_to_audio.py --text-only     # write the script, no audio
Output: paper/audio/paper_narration.txt and paper_narration.wav
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "paper" / "ssrn_diffusion_paper.tex"
CONSTRUCT = REPO / "paper" / "section_construct.tex"
OUT = REPO / "paper" / "audio"

DROP_ENV = ["table", "tabular", "figure", "center", "thebibliography", "equation"]
SPEAK = [
    (r"\\dkl\{\}", " lead "),
    (r"\$t = ([\-0-9.]+)\$", r" t of \1 "),
    (r"\$\\mu = ([0-9.]+)\\%\$", r" \1 percent "),
    (r"\$\\sigma\$", " one standard deviation "),
    (r"\$\\Delta\$AIC", " the AIC difference "),
    (r"\$K\^\{?-\}?\$", " past-facing surprise "),
    (r"\$K\^\{?\+\}?\$", " future-facing surprise "),
    (r"\$r = ([\-+0-9.]+)\$", r" a correlation of \1 "),
    (r"\$R\^2\$", " R squared "),
    (r"([0-9])\\%", r"\1 percent"),
    (r"\\%", " percent"),
    (r"([0-9])pp\b", r"\1 percentage points"),
    (r"\bpp\b", " percentage points"),
    (r"\$\+([0-9.]+)\$", r" plus \1 "),
    (r"\$-([0-9.]+)\$", r" minus \1 "),
    (r"\$([0-9.]+)\$", r" \1 "),
    (r"\\S ?66\(a\)", " section 66a "),
    (r"\\S ?8\b", " section 8 "),
    (r"\\S ?71\b", " section 71 "),
    (r"class\$\\times\$cohort", "class by cohort"),
    (r"class\$\\times\$year", "class by year"),
    (r"\$\\times\$", " by "),
    (r"---", ", "),
    (r"--", " to "),
    (r"``|''", '"'),
]


def strip_tex(s: str) -> str:
    s = re.sub(r"(?m)^\s*%.*$", "", s)
    for env in DROP_ENV:
        s = re.sub(r"\\begin\{" + env + r"\*?\}.*?\\end\{" + env + r"\*?\}",
                   " ", s, flags=re.DOTALL)
    s = re.sub(r"\\(?:includegraphics|label|bibitem|bibliographystyle|input)"
               r"(\[[^\]]*\])?\{[^}]*\}", " ", s)
    s = re.sub(r"\\footnote\{(?:[^{}]|\{[^}]*\})*\}", " ", s)
    s = re.sub(r"\\cite[tp]?(\[[^\]]*\])?\{[^}]*\}", " ", s)
    s = re.sub(r"\\citeauthor\{[^}]*\}", " ", s)
    s = re.sub(r"~?\\ref\{[^}]*\}", " ", s)
    s = re.sub(r"\\url\{[^}]*\}", " the repository ", s)
    s = re.sub(r"\\\[.*?\\\]", " ", s, flags=re.DOTALL)
    for a, b in SPEAK:
        s = re.sub(a, b, s)
    s = re.sub(r"\\(?:emph|textbf|textit|textsc|texttt)\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\(?:noindent|maketitle|appendix|toprule|midrule|bottomrule|"
               r"small|centering|quad|qquad|newpage|clearpage)\b", " ", s)
    s = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^}]*\})?", " ", s)
    s = s.replace("{", " ").replace("}", " ").replace("$", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n\s*\n+", "\n\n", s)
    return s.strip()


def sections(tex: str, include_appendix: bool):
    body = tex
    if not include_appendix:
        i = body.find(r"\appendix")
        if i > 0:
            body = body[:i]
    out = []
    abst = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", body, re.DOTALL)
    if abst:
        out.append(("Abstract", abst.group(1)))
        body = body[abst.end():]
    parts = re.split(r"\\(section|subsection)\*?\{([^}]*)\}", body)
    if len(parts) > 1:
        pre = parts[0]
        if pre.strip():
            out.append((None, pre))
        for k in range(1, len(parts) - 2, 3):
            out.append((parts[k + 1], parts[k + 2]))
    else:
        out.append((None, body))
    return out


def main() -> int:
    include_app = "--all" in sys.argv
    text_only = "--text-only" in sys.argv
    tex = SRC.read_text(encoding="utf-8")
    tex = tex.replace(r"\input{section_construct}",
                      CONSTRUCT.read_text(encoding="utf-8"))

    chunks = ["Vocabulary position and trademark lifecycles. "
              "An event-dated corpus and a lead-lag text measure for the "
              "commercial economy."]
    for title, raw in sections(tex, include_app):
        body = strip_tex(raw)
        if len(body) < 200:
            continue
        if title:
            clean = strip_tex(title)
            chunks.append("Section. " + clean + ".")
        chunks.append(body)

    text = "\n\n".join(chunks)
    OUT.mkdir(parents=True, exist_ok=True)
    txt = OUT / "paper_narration.txt"
    txt.write_text(text, encoding="utf-8")
    words = len(text.split())
    print(f"[text] {words:,} words -> {txt}", file=sys.stderr)
    print(f"[text] about {words/150:.0f} minutes at a normal reading pace",
          file=sys.stderr)
    if text_only:
        return 0

    wav = OUT / "paper_narration.wav"
    ps = f"""
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$v = $s.GetInstalledVoices() | Where-Object {{ $_.VoiceInfo.Name -like '*Zira*' }}
if ($v) {{ $s.SelectVoice('Microsoft Zira Desktop') }}
$s.Rate = 1
$s.SetOutputToWaveFile('{wav}')
$s.Speak([IO.File]::ReadAllText('{txt}'))
$s.Dispose()
"""
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True)
    mb = wav.stat().st_size / 1e6
    print(f"[audio] {wav} ({mb:.0f} MB)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
