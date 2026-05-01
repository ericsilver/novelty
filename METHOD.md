# Method (DeDeo KL update)

The original `README.md` proposes a chronological Bayesian / Shannon-surprise
approach. This file records the methodological update agreed 2026-04-26:
**replace the chronological Bayesian update with KL divergence per Simon DeDeo's
prospective/retrospective surprise framework.**

## Quadrant interpretation

For each filing at time `t`, compute two KL divergences between its
description's content distribution `p_i` and reference distributions estimated
from neighboring time windows:

| Score | Reference window | Reading |
|---|---|---|
| Prospective surprise | `[t - W, t)` | how surprising is the filing given what came before? |
| Retrospective surprise | `(t, t + W]` | how obvious does the filing look once we see what came after? |

Quadrants:

- **Prospectively surprising + retrospectively obvious → innovative.** Novel at
  filing time, then imitated/normalized.
- **Prospectively obvious + retrospectively surprising → tired / last-mover.**
  Looked normal at the time, looks like the end of a played-out trend in
  hindsight.
- Other quadrants: derivative or generic.

## Operational defaults

- Window `W` = 5 years (look-ahead for retrospective, look-behind for prospective).
- Distribution form: **term-frequency multinomial** over the unigram + bigram
  dictionary, as the v0 baseline. LDA topic mixtures (~50–200 topics) are
  retained as a planned replication to guard against artifacts of the BoW model.
- Unit of analysis for downstream RQs: registrant-year.
- KL is computed with smoothing (Dirichlet prior or +α) to avoid `inf` for
  tokens absent from a reference window.

## Burn-in vs. legitimate novelty

A naïve burn-in trims the early years of the corpus because the priors are
undeveloped and everything looks novel. But a **whole new industry being born**
should look highly innovative — that's the signal we want, not noise. So we
take two passes:

1. **Per-class run (Class 032 first).** Class 032 is mature for the entire
   NICE era (post-1973), so a few-year burn-in inside the class is fine.
2. **Universal run, all classes.** Build a corpus across every NICE class, fit
   one shared distribution per window, and use the **first ~5 years of the
   shared corpus** as the burn-in. New-class births are then preserved as
   genuinely high-surprise events, scored against the cross-industry
   background rather than against an immature within-class prior.

Both runs produce per-filing surprise scores. Comparing them is itself an
analysis: if a filing is high-surprise within Class 032 but moderate against
the universal background, the novelty is intra-industry; if both are high, the
novelty is cross-industry.

## Open methodological choices

- WordNet phrasal clustering (per the README) is deferred — start with a plain
  unigram + bigram dictionary; revisit if topics look noisy.
- Lemmatization vs. stemming: defer until vocabulary is built.
- Burn-in years near the corpus start (1984) will be excluded from scored
  output but used to seed reference distributions.
