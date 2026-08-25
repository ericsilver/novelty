"""Run the robustness variant matrix sequentially, without agents.

Each variant copies a base script into scripts/variants/, patches named
constants by regex, redirects its result path into paper/v3/_eval/, and runs
it. Failures are recorded and the runner continues. Outputs:

  paper/v3/_eval/<variant>.json     one result file per variant
  paper/v3/_eval/RUNNER_STATUS.json running status, refreshed after each variant

Usage: python scripts/variants_runner.py [only_prefix]
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SV = REPO / 'scripts' / 'variants'
EV = REPO / 'paper' / 'v3' / '_eval'
PY = REPO / '.venv-analysis' / 'Scripts' / 'python.exe'
SV.mkdir(exist_ok=True)
EV.mkdir(parents=True, exist_ok=True)

# (variant name, base script, list of (regex, replacement) patches, args)
V = []

def add(name, base, patches, args=()):
    V.append({'name': name, 'base': base, 'patches': patches, 'args': list(args)})

# --- gate window and cohort halves on the production scoring (global50 only) ---
# Override the dict by re-assigning after the original, rather than regexing a
# brace-nested literal.
slim = [(r'\ndef log',
         '\nSCORINGS = {"global50": "rolling_surprise_class{c}.parquet"}\n\ndef log'),
        (r'for i in range\(3\):', 'for i in range(len(ks)):'),
        (r'for j in range\(i \+ 1, 3\):', 'for j in range(i + 1, len(ks)):')]
add('gate_window_3595', 'resolution_compare.py',
    slim + [(r'GATE_LO, GATE_HI = 4\.0, 8\.5', 'GATE_LO, GATE_HI = 3.5, 9.0')])
add('gate_window_4580', 'resolution_compare.py',
    slim + [(r'GATE_LO, GATE_HI = 4\.0, 8\.5', 'GATE_LO, GATE_HI = 4.5, 8.0')])
add('cohorts_0209', 'resolution_compare.py',
    slim + [(r'REG_LO, REG_HI = 2002, 2018', 'REG_LO, REG_HI = 2002, 2009')])
add('cohorts_1018', 'resolution_compare.py',
    slim + [(r'REG_LO, REG_HI = 2002, 2018', 'REG_LO, REG_HI = 2010, 2018')])

# --- internet pattern and arrival threshold ---
NARROW = (r'PATTERN = \([^)]*\)',
          'PATTERN = (r"\\\\binternet\\\\b|\\\\bworld wide web\\\\b|\\\\bweb ?sites?\\\\b'
          '|\\\\bwebsites?\\\\b|\\\\be-?commerce\\\\b|\\\\belectronic commerce\\\\b")')
add('internet_narrow', 'internet_breakout.py', [NARROW])
add('convergence_arrival_1pct', 'internet_convergence.py',
    [(r'ARRIVAL_SHARE = 0\.02', 'ARRIVAL_SHARE = 0.01')])
add('convergence_arrival_4pct', 'internet_convergence.py',
    [(r'ARRIVAL_SHARE = 0\.02', 'ARRIVAL_SHARE = 0.04')])

# --- surge parameters ---
add('surge_double_15', 'theme_surge_class.py', [(r'DOUBLE = 2\.0', 'DOUBLE = 1.5')])
add('surge_double_30', 'theme_surge_class.py', [(r'DOUBLE = 2\.0', 'DOUBLE = 3.0')])
add('wave_entry_4', 'wave_timing.py', [(r'ENTRY_YEARS = 6', 'ENTRY_YEARS = 4')])
add('wave_entry_8', 'wave_timing.py', [(r'ENTRY_YEARS = 6', 'ENTRY_YEARS = 8')])

# --- combination parameters (expensive; keep last) ---
add('recomb_bigram_2', 'combination_measures.py',
    [(r'BIGRAM_MIN, UNIGRAM_MIN = 5, 50', 'BIGRAM_MIN, UNIGRAM_MIN = 2, 50')], ['009'])
add('recomb_bigram_10', 'combination_measures.py',
    [(r'BIGRAM_MIN, UNIGRAM_MIN = 5, 50', 'BIGRAM_MIN, UNIGRAM_MIN = 10, 50')], ['009'])
add('combination_thresh_005', 'combination_measures.py',
    [(r'THRESH = 0\.10', 'THRESH = 0.05')], ['009'])
add('combination_class_035', 'combination_measures.py', [], ['035'])


def main() -> int:
    only = sys.argv[1] if len(sys.argv) > 1 else ''
    status = {'started': time.strftime('%Y-%m-%d %H:%M:%S'), 'variants': {}}
    for v in V:
        if only and not v['name'].startswith(only):
            continue
        name = v['name']
        dst = SV / f"{name}.py"
        src = (REPO / 'scripts' / v['base']).read_text(encoding='utf-8')
        # Copies live one level deeper than scripts/, so re-anchor the repo path.
        src = src.replace('parents[1]', 'parents[2]', 1)
        n_applied = 0
        for pat, rep in v['patches']:
            src, k = re.subn(pat, rep, src, count=1, flags=re.S)
            n_applied += k
        if n_applied < len(v['patches']):
            status['variants'][name] = {'ok': False, 'error': f'patch failed ({n_applied}/{len(v["patches"])})'}
            (EV / 'RUNNER_STATUS.json').write_text(json.dumps(status, indent=1))
            print(f'[{name}] PATCH FAILED', flush=True)
            continue
        # Redirect result writes into _eval with the variant name, and repair
        # the input reads that some scripts take from the results directory.
        src = re.sub(r'RES / f?["\']([A-Za-z0-9_]+)(\{[^}]*\})?\.json["\']',
                     lambda m: f'RES / "{name}.json"', src)
        src = src.replace('RES = REPO / "paper" / "results"',
                          'RES = REPO / "paper" / "v3" / "_eval"\n'
                          'BASERES = REPO / "paper" / "results"')
        src = src.replace('CACHE = RES /', 'CACHE = BASERES /')
        src = src.replace('RES / "per_industry_names.json"', 'BASERES / "per_industry_names.json"')
        dst.write_text(src, encoding='utf-8')
        t0 = time.time()
        print(f'[{name}] running...', flush=True)
        r = subprocess.run([str(PY), str(dst), *v['args']], cwd=str(REPO),
                           capture_output=True, text=True,
                           env={**__import__('os').environ, 'PYTHONIOENCODING': 'utf-8'})
        ok = r.returncode == 0
        status['variants'][name] = {'ok': ok, 'seconds': round(time.time() - t0),
                                    'tail': (r.stderr or '')[-800:]}
        (EV / 'RUNNER_STATUS.json').write_text(json.dumps(status, indent=1))
        print(f'[{name}] {"ok" if ok else "FAILED"} in {round(time.time()-t0)}s', flush=True)
    status['finished'] = time.strftime('%Y-%m-%d %H:%M:%S')
    (EV / 'RUNNER_STATUS.json').write_text(json.dumps(status, indent=1))
    print('[runner done]', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
