# Revision plan from the adversarial panel (2026-07-02)

Five hostile referees (identification, measurement, literature, data-integrity,
framing) attacked the paper; every fatal/major challenge was then checked
against the tex by a defense agent. 39 serious findings survived triage: 28
with residual severity **major**, 11 **minor**. Raw panel output preserved in
the session task log; this file is the actionable synthesis.

## The verdict in one paragraph

The panel converged from three independent directions on the same core threat:
**the gate result may be substantially a filer-composition effect, not a
vocabulary effect.** The penalty concentrates in self-filed, use-based
registrations (+18pp) and nearly vanishes among counsel-represented filings in
the cleanest cohort (+1.1pp, 2019) — the exact fingerprint of the post-2015
subsidized foreign mass-filing wave. Simultaneously: the inferential apparatus
(binomial SEs on pooled quintile cuts, no owner clustering, no covariates, no
regression anywhere in the mark-level pipeline) cannot distinguish these
stories. The panel also endorsed a reframe the paper is already halfway to:
the durable contributions are (1) the 13.99M-case event-dated prosecution
dataset, (2) the corpus-scale term-vs-topic measurement hazard, (3) a bounded
lifecycle-risk correlate — not "commercial novelty" per se.

## Tier 1 — blocking analyses (the paper should not be submitted without these)

1. **The composition-vs-construct decider.** Re-estimate the gate result as an
   LPM/logit with class×cohort FE + covariates (log description length, counsel,
   basis, owner cumulative filing count, **owner domicile US/CN/other**) and
   owner-clustered SEs, on unique serials (kill multi-class double counting).
   Report the within-counsel-represented, US-domiciled cohort curve. *This
   single specification decides the paper's headline.* Attorney name, basis,
   and domicile are all in case_extras / the TRTYRAP parse.
2. **Look-ahead purge.** (a) The retrospective window overlaps the outcome
   window → promote a prospective-only ("freshness") variant computable at
   filing date as the primary risk score. (b) Vintage refit of vocabulary +
   LDA (fit on pre-t data only) for at least one diagnostic class (009), show
   the gate lift survives.
3. **Full quintile profiles, both scorings.** The event-dated token-ΔKL
   profile is U-shaped (Q1 fails 5.8pp more than Q3) — flatly inconsistent
   with the "scoring-robust monotone penalty" framing. Publish the profiles as
   figures, decompose into |ΔKL| (unsigned atypicality) and sign components.
4. **C8 code forensics appendix.** Event-code inventory with modal
   descriptions/counts for every cancellation-adjacent code; age-at-C8
   histogram by cohort (a clean gate signature spikes at ~6.0–6.5y);
   partial-cancellation and §8+15 handling. Two brand anecdotes are not
   ground truth in a repo with a documented status-code mismap.
5. **Gate economic content.** Link gate failure to observable commercial
   death: owner's last-ever filing, cessation of owner filing activity, EDGAR
   exit for the matched subset. A 9pp lift on a maintenance form needs a
   welfare referent.

## Tier 2 — reframe and rewrite decisions

6. **Retitle/reposition.** "Commercial novelty" is unearned post-Appendix B
   (the primary scoring provably cannot see combinatorial/lexical/wave novelty).
   Options the panel converged on: lead with corpus + construct + measurement
   hazard — e.g. an introduction paper for the event-dated trademark
   prosecution corpus with a lead/lag vocabulary-position measure and the
   term-scoring hazard as twin results. This matches the actual goal:
   introduce corpus and construct to other researchers.
7. **Engage the literature the intro currently skips**: KPST (patent-text
   novelty), Kalyani (creativity via new terminology), Argente et al.
   (trademarks→product innovation), Dinlersoz-Goldschlag (trademark-firm
   linkage). Marginal claim: same construct family, ported to the full
   commercial economy (firms that never patent), with an event-dated outcome
   the patent corpus lacks.
8. **"Resonance" naming**: either rename (lead/lag vocabulary position;
   vocabulary-drift alignment) keeping DeDeo as inspiration, or implement the
   actual agent-stream object. The corpus-of-strangers is not the
   Murdock/Barron object.
9. **Demote the AI-era verdict** to a worked example of era monitoring with
   sensitivity analysis; strip the macro pronouncement.
10. **Restructure craft**: abstract to ~150 words (one sentence per anchor
    result); withdrawn-results narration → "Changes from earlier drafts"
    appendix; strip point estimates from the agenda section; fix quadrant
    figure (post-2019 marquee examples are outside the declared interpretable
    range — use OpenAI 2016, White Claw 2016, Instagram 2011, iPod 2001).

## Tier 3 — mechanical fixes (do with Tier 1 re-runs)

11. Uniform observation window across cohorts (C8 age < 6.75) to kill the
    differential-censoring critique; flag 2017–19 accordingly.
12. Competing risks: define the gate risk set as marks alive at 5.0y;
    Aalen-Johansen cumulative incidence by quintile as robustness.
13. 2019 replication: registration-month balance check + capped window (the
    elapsed-only conditioning selects early-2019 registrants).
14. Attrition table for n_terms≥3 / thin-reference drops (22% of corpus,
    correlated with the regressor).
15. Owner-excluded references (leave-one-owner-out) robustness on ΔKL.
16. Per-filing ΔKL reliability: bootstrap reference windows; smoothing sweep.
17. Multi-class text: re-score class 009 with class-matched GS text (GS type
    codes carry the class).
18. FDR control within reported families (44-class forest, cohort curve,
    window variants); state the analysis inventory across the three pipeline
    generations.
19. Quote the topic-scored patent-distinctness coefficient in the abstract,
    not the legacy term/decay estimate.
20. Office-action incidence and type by ΔKL quintile (examiner-channel bound
    currently counts only formal adversarial proceedings; ~45% of
    registration failures are abandonments, often post-office-action).
21. Diffusion section: run the 45-class flow build (now mechanical) with a
    volume-adjusted null for origination; carry the four-class caveat into the
    abstract or cut edge counts from it.

## Sequencing

Week 1: Tier 1 items 1–3 (they share one rebuilt estimation table). The
composition decider determines which reframe in Tier 2 is honest — write
nothing until it lands. Week 2: forensics appendix (4), economic content (5),
then the rewrite (6–10). Tier 3 rides along with the re-runs.

## What the panel did NOT break

- The registration inverse-U (survives with within-class caveats).
- Patent-distinctness direction (needs the right coefficient quoted).
- The 2019 replication design itself (needs the balance check).
- The measurement-hazard result (term scoring manufactures firm correlations)
  — repeatedly identified as the paper's strongest publishable claim, though
  it needs the in/out-of-vocabulary decomposition to carry headline weight.
- The corpus itself: the event-dated 13.99M prosecution parse was repeatedly
  called the most durable contribution.
