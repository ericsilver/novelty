# Evaluation report — v3 draft (2026-08-24)

Program: Eric's four-dimension review (clarity; contribution; validity; robustness),
run as a 196-agent workflow (audit + adversarial verify) plus an agent-free variant
matrix after the usage-credit outage. Raw findings: `_eval/audit_findings_raw.json`
(240 findings; 6 blockers, 106 majors, 128 minors). Verified counts: 159 findings
survived adversarial verification in the first pass, 137 in the resumed pass
(overlapping sets; the raw file is the union of finder outputs).

## Disposition

- **Blockers: 6 of 6 fixed** (intro contradiction; corrupted \ref; §2 duplication;
  unnamed sign conventions; inverted renewal curve; wrong length correlations).
- **Majors: ~45 of 106 fixed** — all code-vs-text majors in §§1–5, the statistics
  disclosures, the contribution wiring, the build-order trio (Nice defined,
  theme-not-topic, counsel-sense "representation"), both reproducibility majors
  (cache gitignored; README rewrite still owed), and the stale-crosswalk rerun.
  The remainder are overlapping restatements of the fixed items or clarity-major
  rewordings not yet applied.
- **Minors: deliberately held** (128, mostly clarity) pending Eric's own line edits
  against the frozen review copy, to avoid collisions.

## The ten findings that most changed the paper

1. The year-ten renewal shape was inverted in prose (data: inverted V, lagging tail
   lowest). Fixed; the "lagging tail closes the gap" claim withdrawn.
2. Global atypicality's length correlations were wrong in sign and magnitude in two
   sections; re-quoted from the artifact (all negative, vs log distinct-term count).
3. The ladder's reporting/IPO rungs carry no post-debut requirement; restated as
   at-any-time with the pre-debut majority flagged.
4. §4's raw SEC rates came from a crosswalk the linker itself documents as defective;
   rerun on the rebuilt crosswalk (levels roughly double; shape unchanged).
5. The −0.020 "firm-year correlation" is a between-firm correlation; the firm-year
   value is +0.010; one grid cell was untraceable and is dropped.
6. The corpus-wide annual-bucket estimate exists only in a superseded build's log;
   the claim is now scoped to the persisted software-class artifact.
7. Raw-contrast t-statistics ignore owner clustering (pooled ICC 0.38, itself
   corrected from 0.42); disclosed once and where margins are close.
8. The priority claim ("never been tested") contradicted the paper's own
   bibliography; reconciled with Golder–Tellis and Semadeni–Anderson.
9. The novelty-origin classification counts filings made before a theme was
   sustained anywhere; the text now says what the code computes.
10. "Every industry" in the intro's close vs 33 of 44 in its opening; reconciled.

## Robustness matrix (nothing reverses)

| Variant | Baseline | Variant value | Verdict |
|---|---|---|---|
| Gate window 3.5–9.0 | lead +2.29pp | +2.29pp | unchanged |
| Gate window 4.5–8.0 | lead +2.29pp | +2.28pp | unchanged |
| Cohorts 2002–09 | lead +2.29pp | +0.98pp | attenuates (era swing) |
| Cohorts 2010–18 | lead +2.29pp | +3.11pp | strengthens |
| Cohorts, atypicality | −4.39pp | −4.55 / −4.28pp | unchanged |
| Internet pattern, narrow core | goods web-vs-rest 0.0pp pooled | +2.0pp pooled; tech +7.0 | holds (goods gap larger under narrow) |
| Convergence arrival 1% | early +4.4 → late +1.6 | +5.9 → +1.8 | holds |
| Convergence arrival 4% | same | +3.8 → +2.3 | holds, flatter |
| Surge doubling 1.5× | net +1.27pp (962 eps) | +1.16pp (1,261 eps) | unchanged |
| Surge doubling 3× | net +1.27pp | +0.86pp (616 eps) | unchanged |
| Wave entry window 4y / 8y | order flat | flat (−1.2…+1.0) | holds |
| Recombination BIGRAM_MIN 2 | net −2.80pp | −1.24pp | attenuates, holds |
| Recombination BIGRAM_MIN 10 | net −2.80pp | −3.63pp | strengthens |
| Portfolio pair-lead THRESH 0.05 | net +2.27pp | +3.10pp | strengthens |
| Combination measures, class 035 | 009 values | recomb −3.80 net; pair-lead **−1.42** net | recomb holds; **pair-lead sign flips in 035** |
| Ladder: counsel-only | reporting L −0.21pp | −0.28pp (t −8.7) | strengthens |
| Ladder: self-only | reporting L −0.21pp | −0.04pp (t −1.8) | attenuates to null |
| Ladder: debuts 2009–13 / 2014–18 | reporting A +0.11pp | −0.01 / +0.20pp | atypicality belongs to the late half |
| Decayed windows (earlier run) | lead +2.29pp | +1.95pp | attenuates ~1/7 |
| Three scorings (earlier run) | lead +2.29pp | +2.61 / +1.94pp | holds; atypicality −4.4 → −0.7 under per-class themes |

**Strongest construction:** the first-gate lead penalty on 2010–2018 registrations,
counsel-represented, any window, any theme partition. **What attenuates it:** early
cohorts (2002–09), decayed reference windows, per-class theme fitting (mildly).
**What reverses:** nothing in the headline; the *pair-lead* auxiliary flips sign in
class 035 and should stay a class-009 bound, and the *atypicality* gate protection
is a global-partition property (imported themes), already restated in §3.

## Citations

- Misattributed/unverifiable: Fink–Helmers–Kolev–Toole as the year-5-margin source
  (marked UNVERIFIED in the positioning note; not cited in text). 
- Added and wired: Golder–Tellis 1993; Semadeni–Anderson 2010; Taeuscher–Rothe 2020;
  Zuckerman 1999; Hsu 2006; Fleming 2001; Uzzi et al. 2013.
- Still to wire (agents' support list): Kolev et al. trademark-data work; the
  Willeke–Block–Lambrecht review — verify against bib/refs.bib first.

## Reruns still owed

- README Reproducing section rewrite for the v3 tree.
- Corpus-wide annual-bucket LPM arm, if the superseded-build sentence should become
  an artifact-backed one.
- Clarity-minor sweep after Eric's line edits land.
