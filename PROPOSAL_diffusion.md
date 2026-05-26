# Business-Model Diffusion — a research program

*Innovation as idea diffusion: detecting business-model themes, tracking them as they
transit across industries, and asking how many genuinely new ideas there are, how far
and how fast they travel, who carries them, and what the traffic responds to.*

Status: program proposal, 2026-05. Sibling to `PROPOSAL.md` (original Bayesian-surprise
proposal) and `METHOD.md` (the DeDeo KL update). This document does **not** replace the
∆KL papers; it reuses their measurement apparatus for a different unit of analysis and a
different — and, I will argue, more defensible — construct.

---

## 0. Why this program, and how it relates to the ∆KL papers

The ∆KL papers measure, for each filing, a **scalar** timing position: how off-trend its
goods/services vocabulary is relative to its own NICE class. The unresolved construct
problem there is that "off-trend vocabulary" has several generators (firm foresight,
drafting fashion, category churn, ID-Manual mechanics) that the scalar cannot separate,
so calling the scalar "innovation" overclaims.

This program changes the unit of analysis from **the filing** to **the theme** — a
recurring, semantically coherent vocabulary cluster that names a business-model component
("on-demand peer-to-peer matching," "subscription replenishment," "distributed ledger of
record," "X-as-a-service"). A theme is an *object with a genealogy*: it is born somewhere,
spreads to some industries and not others, at some speed, and eventually saturates or dies.
Studying themes-as-objects buys three things the scalar program cannot:

1. **It sidesteps the construct fight.** Whether or not a reviewer accepts that importing a
   tech pattern into transportation is "innovation," the spread of a describable commercial
   pattern across industries is unambiguously **business-model diffusion**, and that is
   measurable on its own terms. The program therefore has two registers: a robust
   descriptive claim (business-model diffusion) and a contested interpretive claim
   (diffusion = innovation). We lead with the former and reach for the latter only where
   the evidence licenses it.

2. **It converts the steelman's worst objection into the dependent variable.** The ∆KL
   skeptic says high novelty is often just "a hot class churning." Here, *a class churning
   because a theme is arriving* is precisely the event we want to detect. The confound
   becomes the signal.

3. **It is the operationalization the original proposal already wanted.** `PROPOSAL.md`
   line 32: "some technologies are not novel in some industries [but] using those
   technologies within another industry carries a greater degree of novelty… we can track
   the adoption of new methods across industries." `METHOD.md` lines 53–56 already
   distinguishes within-class from cross-corpus surprise. This program is the build-out of
   that sentence.

The canonical intuition (Eric's): for a window of years a great many startups described
themselves as "Uber, but for *X*." A peer-to-peer on-demand dispatch description landing in
**Class 039 (transportation)** in 2010 looks shockingly novel against that class's history;
the *same* description in **Class 042 (software/tech services)** by 2014 is tired
boilerplate. The novelty is **borrowed** — locally new, globally old. The borrowing event,
and the path of borrowings, is the phenomenon.

---

## 1. The construct, stated precisely

**A theme** θ is a coherent cluster of tokens/phrases representing one business-model
component. Themes are *multi-token and semantically coherent by construction* — this is a
deliberate guard (see §4): a single migrating word ("cloud") is not a theme; a recurring
phrase-structure or co-occurring phrase set is.

**Business-model diffusion** is the spread of a theme across NICE classes over time. We
represent it as a **theme × class × time panel** and study its dynamics. The headline
constructs are:

- **Novelty rate** — how many genuinely new themes are born per period (RQ1).
- **Breadth** — how many industries a theme reaches (RQ2).
- **Speed and duration** — how fast it spreads and how long it stays live (RQ3).
- **Provenance** — whether themes are born from / carried by entrants or incumbents (RQ4).
- **Demand sensitivity** — whether diffusion tracks margins (rents) or revenue (market
  size) (RQ5).

**The contested bridge to innovation.** Following Gatignon's novelty-vs-innovation
distinction already adopted in `PROPOSAL.md` (novelty becomes innovation when it is
*replicated*), a theme that diffuses *is* a novelty that succeeded — replication across
industries is the success criterion, observed directly rather than imputed. This is the one
place the innovation interpretation is earned rather than asserted: a theme's breadth-speed
profile is its replication record. We claim "innovation" only for **themes that originated
locally and then diffused**, and only as "a business model novel enough to be copied," not
"a technical invention."

---

## 2. Measurement

### 2.1 Condition on the grant, not the filing

The entire program is conditioned on **registered (granted) marks**, not applications. The
rationale, which deserves its own short paper section:

- A filing is a *strategic option*; a registration is a *completed commitment*. An applicant
  who files broad, speculative, or defensive language and then abandons the mark (no
  Statement of Use, opposed, refused, or simply dropped) reveals that the language reflected
  intent-to-protect, not realized commercial activity. Conditioning on grant strips most of
  that strategic noise.
- Grant therefore raises the signal-to-noise ratio of the vocabulary as a description of an
  *operating* business model. Applications are **not without signal** — abandonment patterns,
  speculative language waves, and refusal reasons are themselves informative about strategic
  behavior versus investment — so a dedicated section will analyze the filing/grant **gap**
  rather than discard it silently. The claim is "grant is the cleaner conditioning event for
  diffusion," not "applications are worthless."
- **Caveat to carry openly (per critic):** grant is itself an outcome of a ~12–18 month
  examination process. Conditioning on it selects on registrability (distinctive, non-merely-
  descriptive language survives; generic language is refused) and on applicant follow-through
  (competent counsel, capitalization). This *selects on the regressor's neighborhood* — the
  very distinctiveness we measure is correlated with surviving examination. The filing/grant
  section must report the direction and magnitude of this selection (compare theme-detection
  on the filed vs. granted text for the same marks) so it is bounded, not assumed away.

### 2.2 ∆KL is the *absolute* difference; the sign is analyzed separately

Per the measurement as built: **∆KL = |KL_pros − KL_retr|** — the *magnitude* of a filing's
timing asymmetry, i.e., how far off-trend in *either* direction it sits. The **signed**
quantity (KL_pros − KL_retr) is analyzed as a separate split and carries the direction:

- **Ahead / resonant** (prospectively surprising, retrospectively obvious): novel at grant
  time, then imitated — the leading edge.
- **Behind / tired** (prospectively obvious, retrospectively surprising): looked normal at
  the time, looks like a played-out trend in hindsight — the trailing edge.

For diffusion this split is not a nuisance — **it recovers a filing's position on the
adoption curve of whatever theme it carries.** Early adopters of a theme in a class score
*ahead*; laggards score *behind*. So the signed ∆KL gives, per filing, a read of "where on
the S-curve does this filing sit for the theme it is carrying" — the micro-foundation of the
theme-level diffusion curves in §2.5. We will validate that signed-ahead filings
systematically precede signed-behind filings within a (theme, class) cell; if they do not,
the curve interpretation fails.

### 2.3 From the scalar to **borrowed novelty**: the within-class vs. cross-corpus split

`METHOD.md` already runs two references: a **within-class** distribution and a **universal
(cross-corpus)** distribution. The diffusion signal lives in the *gap* between the two
surprises for the same filing:

| within-class novelty | cross-corpus novelty | reading |
|---|---|---|
| high | high | **genuinely new theme** (born here, new everywhere) — candidate origin event |
| high | low | **borrowed novelty** — locally new, globally old → a *transit/adoption* event |
| low | high | measurement artifact / class-specific jargon that is corpus-rare (inspect) |
| low | low | derivative within an established theme |

The **(high within-class, low cross-corpus)** cell is the operational definition of an
"Uber-for-X" transit at the filing level. Aggregated to the theme level it yields the
diffusion path. This is the single most important measurement object in the program and it
falls directly out of the two reference runs already specified.

### 2.4 Theme extraction and its validation

v0 candidate generators, to be run in parallel and reconciled:

- **Topic model** (LDA, 50–200 topics, per `METHOD.md`) over the unigram+bigram dictionary.
- **Embedding-cluster** of phrases (sentence-embed the g/s spans, cluster, label).
- **High-PMI collocations / statistically improbable phrases** (the SIP idea in
  `PROPOSAL.md` Appendix 1).

A candidate is promoted to a **theme** only if it is **stable across ≥2 generators**
(token-set Jaccard above a pre-registered threshold) and survives **human face-validation**
on a random sample (target ≥ 0.8 rater agreement that the cluster names a recognizable
business-model component). Themes that exist in only one method are reported as fragile and
excluded from headline results. This is the firewall against theme reification (§4).

### 2.5 The theme × class × time panel and its primitives

For each (theme θ, NICE class c, period t) compute **prevalence** = share of class-c grants
in t that carry θ above loading threshold τ. From this panel:

- **Birth.** t₀(θ) = first period corpus-wide prevalence clears a noise floor; c₀(θ) = the
  origin class (earliest above-floor class). Simultaneous births and multi-class first
  appearances are flagged and hand-adjudicated for the headline theme set.
- **Transit event.** For c ≠ c₀, arrival ta(θ, c) = first above-floor period in c. The
  ordered set {(c, ta)} is the **diffusion path**.
- **Breadth.** B(θ) = count of classes ever above floor (0–45). Expect a fat-tailed
  distribution: most themes reach few classes; a few become general-purpose.
- **Speed & shape.** Fit a **Bass (1969) diffusion model** to cumulative adoption — either
  across classes N_c(θ,t) or across filings N_f(θ,t) — yielding innovation coefficient *p*,
  imitation coefficient *q*, and ceiling *m*. Simpler companions: time-to-birth→50%-breadth
  (t₅₀), peak-adoption period.
- **Duration / death.** Active window = birth → last new-class arrival; death = sustained
  prevalence decline. Distinguish *saturation* (reached its classes, still used) from *death*
  (abandoned).

### 2.6 Provenance typology — who originates, who carries

For birth filings and early arrivals in each destination class, classify the registrant on
two axes using the owner panel (`owner_address.parquet`, prior grants):

- **Incumbent vs. debut** — does the registrant have prior grants?
- **Carrier vs. adopter** — was the registrant already active in the *origin* class (a
  carrier importing the theme into a new class) or native to the *destination* class (an
  adopter taking on an external idea)?

Crossing these gives a four-way provenance tag — incumbent-carrier, incumbent-adopter,
entrant-carrier, entrant-adopter — that operationalizes the Schumpeter Mark I (entrant-led)
vs. Mark II (incumbent-led) debate at the level of individual diffusion events (RQ4).

---

## 3. Research questions

**RQ1 — How many genuinely novel ideas are there?**
Rate of theme births per year, after filtering noise/proper-nouns/ID-Manual codifications.
The sharper version: does the count of *distinct* themes grow **sub-linearly** in the number
of filings (a Heaps'-law exponent < 1)? If most filings recombine existing themes and
genuinely new themes are rare and not accelerating, that is a quantitative answer to "there
are only so many new ideas; the rest is recombination" — and it speaks directly to Bloom et
al. (2020), *Are Ideas Getting Harder to Find?*, from a wholly independent data source.
Headline statistic candidate: **share of grants that introduce no new theme** (pure
recombination) vs. share that originate one.

**RQ2 — Breadth: how far does an idea expand?**
The distribution of B(θ), and what predicts it: origin class, originator type (RQ4), birth-
era novelty (§2.3), initial *q/p* ratio. Hypothesis: themes born in **central** classes
(software/business-services, NICE 035/042) reach more classes than themes born in peripheral
classes — a trademark analogue of Hidalgo–Hausmann **relatedness / product-space centrality**
(2009). We can build the empirical "business-model space" as the class-to-class theme-flow
network and locate each class by centrality.

**RQ3 — Speed and duration.**
Distribution of Bass (*p, q, m*). Are *later-born* themes adopted faster than earlier ones
(diffusion accelerating over the sample — "idea superhighways"), or slower (crowding)?
Duration: are fast-diffusing themes also fast-dying (fads) or do they persist (durable
platforms)? The fad-vs-platform split is itself a classification worth publishing.

**RQ4 — Who originates and who carries?**
Provenance shares (§2.6) over time and across classes. Two specific tests: (a) are new themes
disproportionately *born* from debut/entrant filings (Schumpeter I) or incumbent
diversification (Schumpeter II)? (b) When a theme crosses an industry boundary, is it carried
predominantly by **entrant-carriers** (firms from the origin class entering the destination)
or adopted by **incumbent-adopters** of the destination class? The boundary-crossing
mechanism is, to my knowledge, not measured at scale anywhere; this is a candidate headline.

**RQ5 — Is diffusion more responsive to margin or to revenue?**
The demand-pull (Schmookler 1966: invention follows *market size / revenue*) vs.
appropriability/rents (innovation follows expected *margins*) debate. Operationalize: model
theme **inflow rate** into class c on c's lagged margin and lagged revenue growth (class-level
financials mapped from the existing financial panel). This is the most ambitious and least
identified question — see §4; expect to deliver *suggestive* evidence and a credible design,
not a clean causal estimate, in the first pass.

**RQ6 — Does borrowed novelty pay? (bridge back to the main papers)**
Does carrying an *early-stage transiting* theme (high borrowed-novelty, signed-*ahead*, before
saturation) predict registrant survival/financials better than within-class novelty alone?
If borrowed-novelty timing predicts outcomes where raw ∆KL does not, that both validates the
diffusion construct and rehabilitates a *localized* innovation claim for the ∆KL papers — the
isolated component the prior critique demanded.

---

## 4. Identification and threats (read before believing any result)

1. **Theme reification.** Clusters can be artifacts of the extraction method. Guard:
   cross-method stability + human validation (§2.4) + a **date/class shuffle null** — randomize
   the (theme, class, time) assignments and confirm real themes show coherent directional
   diffusion (origin → spread, ahead-before-behind) that shuffled data do not. If real and
   shuffled are indistinguishable on the diffusion metrics, the unit is an artifact and the
   program fails here.

2. **Vocabulary migration ≠ idea migration.** A token can enter a new class via generic drift,
   homonymy, or branding fashion with no business-model transfer. Guard: themes are multi-token
   and coherence-validated; single-token "migrations" are excluded by construction. A transit
   event requires the *phrase-structure*, not a word.

3. **NICE classes are legal goods/services buckets, not industries.** 45 trademark categories
   only proxy industry. Multi-class filings and same-firm refilings across classes can
   masquerade as diffusion. Guard: dedupe within registrant-family before counting a transit;
   report results both raw and "first-carrier-only"; cross-walk NICE → NAICS where possible and
   show the diffusion findings survive the coarser industry partition.

4. **Censored births.** A theme's first *trademark* appearance is not its birth in the economy;
   pre-1984 and never-trademarked ideas are invisible. RQ1 measures the rate of new *trademarked
   business-model vocabularies*, a selected subset. State this; do not let "how many ideas
   exist" inflate past "how many surface in registered marks."

5. **Self-inclusion / cohort.** A destination class's reference window includes the transiting
   cohort, so a wave looks both novel and adopted. For diffusion that is fine (we want the wave),
   but for *first-carrier* dating (RQ4) it matters — pin first movers with leave-one-out
   references so a filing is never scored against itself or its same-quarter siblings.

6. **Grant selection** (§2.1) — bounded empirically, not assumed away.

7. **RQ5 is barely identified.** Reverse causality (a theme raises a class's margin, then we
   "find" margin attracts themes), confounding (hot classes have high margin *and* high
   adoption), and ecological inference (class-level financials vs. filing-level themes) all bite.
   Honest first pass: lagged associations with class and year fixed effects, explicitly labeled
   associational. A credible causal design would exploit a **margin/revenue shock exogenous to
   vocabulary** — a tariff, a regulatory opening, a commodity-price move that hits a destination
   class — and ask whether theme inflow responds, and whether it responds to the margin channel
   or the revenue channel of the shock. Flag this as a second-paper design, not a first-pass
   claim.

---

## 5. What would falsify the program

- **No coherent diffusion above the shuffle null** (threat 1) → the construct is noise; stop.
- **Themes don't replicate across extraction methods** → unit is an artifact; stop or rebuild.
- **Borrowed-novelty (high within / low cross) predicts the same outcomes as plain within-class
  novelty** → the within/cross decomposition adds nothing; the diffusion lens is decorative.
- **Signed-ahead filings do not systematically precede signed-behind filings within (theme,
  class) cells** → the adoption-curve interpretation of the sign is wrong; the Bass framing is
  unsupported.
- **Theme-birth count grows ~linearly with filings with no fat tail in breadth** → there is no
  small set of general-purpose ideas diffusing; the "few real ideas, much recombination" thesis
  (RQ1) is false and should be reported as such.

A program that names its own kill conditions up front is the point of doing this under the
critic's stance.

---

## 6. Phased execution

**Phase 0 — runnable now, no new data.**
- Add the cross-corpus reference run alongside the per-class run (already specified in
  `METHOD.md`) and compute the within/cross gap (§2.3) on granted marks.
- Build the theme × class × time prevalence panel from v0 topics.
- Produce the §5 shuffle null and the signed-ahead-precedes-behind check. *If these fail, we
  learn it cheaply before investing in theme curation.*

**Phase 1 — theme curation and the descriptive program.**
- Cross-method theme stability + human validation (§2.4).
- Birth/transit/breadth/speed/duration primitives; the class-to-class flow network and
  relatedness/centrality (RQ2).
- Provenance typology from the owner panel (RQ4). RQ1 Heaps'-law estimate.

**Phase 2 — the contested and the causal.**
- RQ6 outcome bridge (does borrowed novelty pay?) — the link back to the main papers.
- RQ5 margin-vs-revenue: associational first pass, then the shock-based design.
- The filing/grant gap section (§2.1) as its own analysis of strategic vs. realized behavior.

---

## 7. Positioning and literature

- **Diffusion of innovations:** Bass (1969); Rogers. The *p/q/m* vocabulary is the natural
  formal home for breadth/speed/duration.
- **Recombinant growth & atypical combinations:** Weitzman (1998, QJE); Uzzi, Mukherjee,
  Stringer & Jones (2013, Science). Borrowed novelty = recombination across industry
  boundaries, observed directly in commercial-description text.
- **Product space / relatedness:** Hidalgo & Hausmann (2009, PNAS) and the economic-complexity
  program. The class-to-class theme-flow network is a "business-model space"; centrality there
  predicts breadth (RQ2).
- **Are ideas getting harder to find?** Bloom, Jones, Van Reenen & Webb (2020, AER). RQ1 offers
  an independent, text-native estimate of idea-generation rates.
- **Demand-pull vs. rents:** Schmookler (1966) vs. the appropriability tradition — RQ5's framing.
- **Schumpeter Mark I vs. II:** entrant- vs. incumbent-led innovation — RQ4's framing.
- **The resonance apparatus we reuse:** Murdock, Allen & DeDeo (2017, Cognition); Barron,
  Huang, Spang & DeDeo (2018, PNAS) — with the standing caveat that those corpora are
  authored-by-the-novel-agent (notebooks, speeches), whereas g/s text is drafted for protection
  scope; the diffusion register tolerates that mismatch better than the firm-foresight register,
  because diffusion is a property of the *population* of filings, not of any single drafter's
  cognition.
- **Trademark data for economic dynamics:** Dinlersoz et al. (2018) and the USPTO trademark
  case-files tradition for coverage benchmarking.

This is positioned as a **modest, descriptive-first contribution**: a method for detecting
business-model themes and a first map of how they diffuse across industries. The innovation
interpretation is offered as a *bounded* extension (RQ6), not the load-bearing claim.

---

## 8. Open decisions (need a ruling before Phase 1)

- **Period granularity** — year vs. quarter for the panel? (Quarter sharpens speed estimates
  but thins per-class cells.)
- **Theme loading threshold τ and the prevalence noise floor** — pre-register, or tune on a
  held-out class and freeze.
- **Registrant-family resolution** — how aggressively to merge owners (subsidiaries, name
  variants) before counting a transit as cross-firm rather than same-firm refiling.
- **Bass vs. nonparametric** — commit to Bass for *p/q/m*, or fit nonparametric adoption curves
  and report Bass only where it fits? (Many themes will not be Bass-shaped.)
- **NICE → NAICS crosswalk** — build it now (enables RQ5 financial linkage and threat-3
  robustness) or defer?
- **Single program or split?** Whether business-model diffusion ships as its own paper distinct
  from the ∆KL firm-outcome papers and the dynamism/ethnic work — recommended, since the
  diffusion construct does not depend on the contested ∆KL-as-innovation claim and should not
  inherit its review risk.
