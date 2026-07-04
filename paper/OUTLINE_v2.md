# Paper v2 outline — corpus + construct introduction

Post-adversarial-review restructure (2026-07-03). Decisive regression ran:
the gate penalty **survives** the composition attack. Honest headline is
+2.2–2.6pp per within-class×cohort quintile contrast (owner-clustered), not
the naive +3.9pp (~40% of which was class/cohort composition).

## Decisive regression results (paper/results/gate_decisive_regression.json)

3.10M unique serials (deduped from 4.12M class-rows), regs 2002–2018, LPM,
class×cohort FE, SEs clustered on normalized owner (1.27M owners):

| Spec | Q5−Q1 | t |
|---|---|---|
| 1 FE only | +2.22pp | 9.3 |
| 2 FE + controls (counsel, ITU, log len, owner count, domicile, 44e/66a) | +2.28pp | 7.5 |
| 3 counsel-represented only | +2.13pp | 5.8 |
| 4 counsel + US-domiciled | **+2.60pp** | 5.9 |
| 5 self-filed only | +2.45pp | 10.1 |
| 6 counsel+US 2002–07 / 2008–12 / 2013–18 | +0.92 / +3.70 / +3.09pp | 2.4 / 5.0 / 7.0 |
| 7 continuous z, full + controls | +0.87pp/σ | 9.8 |
| 8 continuous z, counsel+US | +0.94pp/σ | 7.3 |

Quintile profile is monotone within cells (+0.44/+0.90/+1.33/+2.28).
Key reversals of the panel's fears:
- The raw self-vs-counsel contrast (18 vs 1pp) was base-rate + composition;
  within-cell gradients are comparable (2.5 vs 2.1pp).
- The modern emergence (2008+) **survives inside the counsel+US stratum**, so
  it is not the foreign mass-filing wave.
Data note: 66a (Madrid) registrations show −46pp on C8 — they file §71, not
§8; exclude or flag them in the final spec (currently controlled).

## Title direction

Drop "commercial novelty" as the primary frame. Working options:
1. "Vocabulary position and trademark lifecycles: an event-dated corpus and a
   lead/lag text measure for the commercial economy"
2. "Where the language of commerce is going: measurement and validation of
   vocabulary lead/lag in 14 million US trademark filings"

## Abstract skeleton (~150 words)

One sentence each: (1) corpus — 13.99M USPTO case files re-parsed into
event-dated prosecution histories, 1990–2024, with per-filing text scores;
(2) construct — a prospective/retrospective KL position measure on
goods/services text, scored on topic distributions, patent-orthogonal,
covering the ~99% of firms that never patent; (3) validation — forward-
leaning registrations fail their first §8 maintenance gate +2.2–2.6pp per
quintile within class×cohort, owner-clustered, robust in the counsel-
represented US-domiciled stratum, emerging post-2008 and replicated
out-of-sample in the 2019 cohort; (4) hazard — term-level scoring
manufactures firm-performance correlations from drafting style that dissolve
under distribution scoring: a caution for the text-as-data innovation
literature; (5) application — cross-class vocabulary diffusion with Bass
dynamics.

## Section-by-section

### 1. Introduction (rewrite)
- Open with the coverage gap: patent-text novelty measures (KPST 2021;
  Kalyani 2025) see invention by the ~1% of firms that patent; trademarks
  time-stamp *commercialization* by the full economy (Dinlersoz-Goldschlag et
  al.; Argente et al.). What's been missing: an event-dated outcome and a
  filing-level text measure on that corpus.
- Three contributions, stated in order of durability: corpus, construct +
  bounded validation, measurement hazard.
- The bounded reading up front: lead/lag vocabulary position, not "novelty."
  DeDeo framework as inspiration; note explicitly this is a corpus-reference
  object, not the agent-stream resonance of Murdock/Barron.

### 2. The corpus (promote to a full section — this is a contribution)
- TRTYRAP re-parse: case_events (prosecution events w/ dates), case_extras
  (counsel, bases, key dates), owner addresses; event-code dictionary.
- **C8 forensics appendix feeds this section** (owed): code inventory,
  age-at-C8 histogram spike at 6.0–6.5y, §8 vs §71 (Madrid) handling,
  partial cancellations.
- Descriptives: filings/registrations by class, cohort, basis, domicile,
  counsel; the post-2015 filing surge documented, since it motivates the
  composition-robust design.
- Release plan: derived per-filing panel (serial, scores, events, covariates)
  as the public data artifact.

### 3. The construct
- Prospective/retrospective KL on topic distributions; window choice; burn-in;
  vintage-fit check (owed: 009 vintage refit); reliability (owed: bootstrap
  CI on per-filing ΔKL); owner-excluded references robustness (owed).
- Appendix B (encodings: Amazon/Kombucha/Ethereum) reframed as *scope
  definition*: what topic scoring can and cannot see — coverage vs partition.
- Full quintile profiles under both scorings (owed: the U-shape exhibit),
  |ΔKL| vs sign decomposition.

### 4. Validation I: the maintenance-gate result (rebuilt around the new table)
- Headline = the decisive regression table (specs 1–8 above).
- Own the correction: naive pooled quintiles gave +3.9pp; ~40% was
  composition; the design-based estimate is +2.2–2.6pp.
- Modern emergence within counsel+US (+0.9 → +3.7 → +3.1pp) as the
  substantive finding about the post-2008 filing economy.
- 2019 out-of-sample replication (uniform-window version — owed: re-run with
  capped C8 age and balance check).
- Economic content (owed): gate failure → owner's last-ever filing / filing
  cessation / EDGAR exit.
- Competing risks + uniform-window robustness (owed, mechanical).

### 5. Validation II: registration completion + patent orthogonality
- Registration inverse-U (within-class version), bounded reading
  (completion ≈ follow-through to use).
- Patent orthogonality quoting the topic-scored coefficient.

### 6. The measurement hazard (promote — strongest methods claim)
- Term-scored firm results (margins, listing) and their dissolution under
  distribution scoring at every T; length-band forensics; in/out-of-vocab
  decomposition (owed).
- Framed as a general caution for text-as-data novelty measures, with the
  trademark corpus as the demonstration at scale.

### 7. Application: cross-class vocabulary diffusion
- 45-class flow build (owed; currently 4-class) with volume-adjusted
  origination null; Bass fits with selection caveats in-line.
- AI-vs-internet as a *worked example* of era monitoring, sensitivity shown,
  no macro verdict.

### 8. What this construct is not (short, replaces scattered demotions)
- Not agent-level resonance; not novelty in the combinatorial sense (Appendix
  B); not a firm-performance predictor (Section 6 is the reason).
- "Changes from earlier drafts" appendix absorbs the withdrawal narrative.

## Owed-work checklist (maps to revision plan tiers)

Blocking: C8 forensics appendix; look-ahead purge (prospective-only variant +
009 vintage refit); U-shape profiles + sign decomposition; gate economic
content; 66a exclusion re-run.
Riding along: uniform windows, competing risks, 2019 balance check, attrition
table, owner-excluded references, reliability bootstrap, per-class GS text
(009), FDR within families, office-action tabulation, 45-class diffusion.
Craft: 150-word abstract, intro literature engagement, quadrant figure
in-range examples, agenda de-numbered.
