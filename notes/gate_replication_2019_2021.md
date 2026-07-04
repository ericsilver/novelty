# Gate-cohort replication: registrations 2019–2021

Script: `scripts/event_gates_2019_2021.py` ·
Results: `paper/results/event_gates_2019_2021.json`,
`paper/results/event_gate_cohort_curve_extended.png` ·
Data cut (max observed C8 event): **2026-04-02**.

## Motivation

The paper's event-dated first-§8-gate penalty (forward-leaning topic-ΔKL
registrations fail their first maintenance gate more often) is estimated on
registration cohorts 2002–2018. The known vulnerability: the most informative
recent cohorts (2016–2018) face their gates in 2021–2024, with the pandemic in
the window. This replicates the estimate on the first post-2018 cohorts using the
identical pipeline (`failed1` = a `C8..` Section-8 cancellation at registration
age 4.0–8.5y).

## Observability

The first §8 gate falls at registration age ~6 (5th–6th anniversary, grace to
6.5); the `C8..` cancellation posts after the grace period. A cohort is
"fully elapsed" only when `reg_date + 6.5y ≤ data cut`. With the cut at 2026-04:

| Cohort | n | median obs. age | share gate-elapsed | base fail₁ | status |
|---|---|---|---|---|---|
| 2019 | 416,492 | 6.73y | 0.75 | 0.331† | **evaluable** |
| 2020 | 319,160 | 5.83y | 0.00 | 0.000 | censored |
| 2021 | 104,394 | 4.87y | 0.00 | 0.000 | censored |

†cohort base is diluted by later-2019 registrations not yet elapsed; the
elapsed-only subset (obs age ≥ 6.5) has base 0.439, matching the historical
~0.45.

2020 and 2021 cannot be evaluated yet — their first gates fall in 2026–2027 and
the cancellation events have not posted. Re-run after a ~2027–2028 USPTO refresh.

## Result — 2019 cohort, gate fully elapsed (n = 313,725)

| Measure | 2019 replication | 2018 (paper) | 2002–2018 pooled |
|---|---|---|---|
| Base fail₁ | 0.439 | 0.455 | 0.458 |
| **Topic-ΔKL lift Q5−Q1** | **+9.11pp ± 0.55** | +9.14pp | +3.94pp |
| Token-ΔKL lift Q5−Q1 | +9.22pp ± 0.58 | — | — |
| Classes positive | 21 / 30 | — | 31 / 44 |

Strata (elapsed 2019 subset) reproduce the paper's buffering structure:

| Stratum | n | base | lift Q5−Q1 |
|---|---|---|---|
| Self-filed (no counsel) | 77,488 | 0.658 | **+18.04pp** |
| Counsel-represented | 236,237 | 0.368 | +1.09pp |
| Use-based | 212,900 | 0.411 | +12.48pp |
| Intent-to-use | 100,825 | 0.499 | +4.44pp |

## Reading

The forward-leaning first-gate penalty persists at full strength in the first
registration cohort whose maintenance gate falls entirely after the pandemic
(2019 regs → 2024–25 gate): +9.1pp, statistically indistinguishable from the
2018 cohort and concordant across topic and token scoring. The counsel and
use-basis buffers replicate. The penalty is therefore not an artifact of the
2016–2018 window. The 2020–2021 cohorts are pending data, not null.
