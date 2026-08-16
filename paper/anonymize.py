"""Produce the double-blind submission build from the identified manuscript.

Research Policy reviews double-blind, so the submitted PDF must not carry the
author's name, affiliation, contact address, acknowledgements, or the
repository URL -- the last of which identifies the author as plainly as the
byline, since the account name is in the path.

This writes paper/ssrn_diffusion_paper_blind.tex beside the original rather
than editing it, so the identified version stays the one of record and the two
cannot drift: the blind copy is regenerated from the original on every run.

Citations to Barron et al. and Murdock et al. are left alone. They are the
construct's source literature, cited as any reader would cite them, and
suppressing them would damage the paper without concealing anything.

Usage:  python paper/anonymize.py
Output: paper/ssrn_diffusion_paper_blind.tex   (then run pdflatex on it twice)
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

PAPER = Path(__file__).resolve().parent
SRC = PAPER / "ssrn_diffusion_paper.tex"
DST = PAPER / "ssrn_diffusion_paper_blind.tex"

REPO_URL = "https://github.com/ericsilver/tm-vocabulary"
BLIND_NOTE = (
    r"\author{}"
    "\n"
    r"\date{}"
)


def main() -> int:
    s = SRC.read_text(encoding="utf-8")

    # 1. Byline, affiliation, contact and acknowledgements all live in one
    #    \author{...\thanks{...}} block. Replace the whole block.
    # A lambda replacement, because backslashes in the substitution text would
    # otherwise be read as regex escapes.
    s, n_auth = re.subn(r"\\author\{Eric Silver\\thanks\{.*?\}\}",
                        lambda _m: BLIND_NOTE, s, flags=re.DOTALL)

    # 2. The repository URL identifies the author through the account name.
    #    Matched loosely, since the abstract's line wrapping shifts with edits.
    s = re.sub(
        r"Code,\s*\npanels, and the corpus build are public at\s*\n"
        r"\\url\{[^}]*\}\.",
        lambda _m: ("Code, panels, and the corpus build are public; the "
                    "repository is withheld for review and will be cited in "
                    "the final version."),
        s)
    s = s.replace(
        f"\\url{{{REPO_URL}}}, indexed in the repository",
        "a public repository, withheld for review, indexed in its")
    n_url = s.count(REPO_URL)

    # 3. Anything else naming the author.
    for bad in ("Eric Silver", "epsilver@gmail.com", "ericsilver"):
        if bad in s:
            print(f"  WARNING: {bad!r} still present after substitution",
                  file=sys.stderr)

    s = s.replace(r"\maketitle",
                  "\\maketitle\n\\begin{center}\\small\n"
                  "Submitted for double-blind review. Author, affiliation, "
                  "acknowledgements and\\\\ the data/code repository are "
                  "withheld and will be restored on acceptance.\n"
                  "\\end{center}")

    DST.write_text(s, encoding="utf-8")
    print(f"[blind] wrote {DST.name}")
    print(f"  author block replaced: {n_auth}")
    print(f"  repository URLs remaining: {n_url}")
    if n_auth != 1 or n_url != 0:
        print("  CHECK THIS BUILD BY HAND before submitting", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
