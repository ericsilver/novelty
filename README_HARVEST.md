# The harvest tail: negative ΔKL pays — and we don't fully understand why

This is a companion note to [`README.md`](README.md) and to the
$\Delta KL$ paper [`paper/main.pdf`](paper/main.pdf). It documents
the single most surprising finding in the corpus.

## The setup, in one paragraph

$\Delta KL = KL_{\text{pros}} - KL_{\text{retr}}$ is the directional
component of a filing's vocabulary surprise. **Positive $\Delta KL$**
means the filing's goods/services prose is unusual against the *recent
past* of its NICE class but resembles the *near future* — the
trend-setter. **Negative $\Delta KL$** means the opposite: prose that
looks like the recent past (typical) but unusual against the near
future (legacy) — vocabulary that the field is leaving behind. The
hypothesis going in, from the optimal-distinctiveness literature
(Deephouse, 1999; Zhao, Fisher, Lounsbury & Miller, 2017), is that
positive $\Delta KL$ should pay. It does.

**The surprise is that negative $\Delta KL$ pays too — and at
observable magnitudes, it pays *more*.**

## The fits, with sign-checked numbers

On the **debut-only panel** ($n = 5{,}088{,}700$ owner-first filings
across all 45 NICE classes), we fit a saturating exponential
$$P(\text{survived 5y}) \;=\; a + b\,e^{-c\,|\Delta KL|}$$
separately on each side of zero.

| Side | $a$ | $b$ | $c$ | $R^2$ on 20 binned points | Pearson $r$ on raw |
|---|---|---|---|---|---|
| Harvest ($\Delta KL < 0$) | $0.573$ | $-0.137$ | $3.33$ | **$0.978$** | $+0.054$ on $\mid\Delta KL\mid$ |
| Innovation ($\Delta KL > 0$) | $0.995$ | $-0.550$ | $0.063$ | $0.499$ | $+0.014$ on $\Delta KL$ |

The harvest fit is *much* tighter ($R^2 = 0.978$ vs.\ $0.499$),
saturates much faster ($c = 3.33$ vs.\ $0.063$, a $52\times$ ratio in
curvature), and approaches its asymptote ($0.573$) well within the
observed range. The innovation fit's asymptote ($0.995$) is at the
upper bound of the fit and is mostly fictional — at $\Delta KL = 1.0$
the curve has only reached $0.479$.

### What that means at observable magnitudes

Plug in concrete $|\Delta KL|$ values:

| $\mid\Delta KL\mid$ | $P(\text{survived})$, harvest | $P(\text{survived})$, innovation | Difference |
|---|---|---|---|
| $0.00$ | $43.6\%$ | $44.5\%$ | $-0.9$ pp (innovation barely ahead at the centre) |
| $0.10$ | $47.4\%$ | $44.9\%$ | $+2.6$ pp |
| $0.20$ | $50.2\%$ | $45.2\%$ | $+5.0$ pp |
| $0.50$ | $54.7\%$ | $46.2\%$ | $+8.5$ pp |
| $1.00$ | $56.8\%$ | $47.9\%$ | $+8.9$ pp |

**At every observable magnitude of $|\Delta KL|$ above $\sim 0.05$,
the harvest side has a larger survival lift than the innovation
side.** At $|\Delta KL| = 0.5$ (a meaningful but not extreme bin),
harvest-side debut filings survive at $54.7\%$ vs.\ $46.2\%$ for
innovation-side debut filings — a $+8.5$ percentage-point spread in
favour of the harvest tail.

### On the full corpus (not debut-only)

The same direction holds. From `paper/results/outcome_by_kl_lines_v2.csv`,
joint $5y$ survival (reached registration **and** still live)
computed as $P(\text{reached}) \times P(\text{live}\mid\text{reached})$
per $\Delta KL$ bin:

| Bin | $P(\text{reached})$ | $P(\text{live}\mid\text{reached})$ | Joint $P(\text{survived 5y})$ |
|---|---|---|---|
| Most-negative $\Delta KL$ (harvest extreme) | $54.8\%$ | $80.8\%$ | **$44.3\%$** |
| Flux-neutral middle | $44.9\%$ | $82.3\%$ | $37.0\%$ |
| Most-positive $\Delta KL$ (innovation extreme) | $47.5\%$ | $87.0\%$ | $41.3\%$ |

The harvest extreme is **the highest** on joint survival across the
entire corpus, beating both the innovation extreme and the
flux-neutral middle.

Note the asymmetric mechanism:
* The **harvest-side lift mostly comes from registration**: harvest
  filings clear the examiner at $54.8\%$ vs.\ $47.5\%$ on the
  innovation side and $44.9\%$ at the middle.
* The **innovation-side lift mostly comes from post-registration
  survival**: conditional on registration, innovation-side marks
  survive at $87\%$ vs.\ $80.8\%$ for harvest-side marks.
* The two lifts compose differently. At the harvest extreme the
  registration boost (high) is partly offset by a lower conditional
  survival (low); at the innovation extreme the registration boost is
  small but conditional survival is excellent.

There is also a **third asymmetry**, in the public-listing signal:
$P(\text{owner ever in SEC EDGAR})$ is **monotone *decreasing* in
$\Delta KL$** at the filing level ($5.78\%$ at the most-negative
decile, $4.0\%$ at the middle, $3.49\%$ at the most-positive). The
harvest tail captures the IPO/listing pattern far more cleanly than
the innovation tail does at the unit of one filing. We do not have a
single-mechanism explanation for why late-stage-imitation filings
are so much more strongly associated with the public-listing outcome
than novelty filings are.

## Sign check, for the paranoid

The reader who suspects we have flipped a sign somewhere can verify
directly from the panel CSV:

```bash
# 1. Pull the binned-by-ΔKL survival rates
python -c "
import polars as pl
df = pl.read_csv('paper/results/outcome_by_kl_lines_v2.csv')
sub = df.filter((pl.col('outcome')=='reached_registration')
                & (pl.col('axis')=='\$\\\\Delta KL\$')).sort('mid')
print(sub.select(['mid','rate']).to_pandas().to_string(index=False))
"
# Output: most-negative mid (-0.476) has rate 0.548; middle has rate ≈0.449.
# Positive ΔKL ↔ more novel against past ↔ trend-setter.
# Negative ΔKL ↔ vocabulary recedes from corpus ↔ harvest tail.

# 2. Verify the saturating fit recovers the same monotonicity
python -c "
import numpy as np
a_n, b_n, c_n = 0.5729, -0.1374, 3.328
for x in [0.0, 0.1, 0.5, 1.0]:
    print(f'|ΔKL|={x:.2f}: survival={a_n + b_n*np.exp(-c_n*x):.4f}')
"
# 0.4355 → 0.4744 → 0.5469 → 0.5680.  Monotone-up in |ΔKL|.  Sign correct.
```

The sign convention is set in `src/novelty/surprise.py` where
`dkl = prospective_kl - retrospective_kl`. KL divergences are
nonneg; `dkl` is the signed residual.

## Why the cause is unclear

Three candidate mechanisms for the harvest-side lift, none of which
the trademark record can separate:

1. **Intelligent harvesting.** Firms that file marks on receding
   vocabulary are making a deliberate bet on a niche the rest of the
   market is leaving. The vocabulary recedes for everyone else, so
   the harvest filer faces fewer collisions on the examiner side and
   fewer competitive disturbances post-registration. The high
   registration rate fits this reading.

2. **Passive late-occupation.** Some firms just keep doing the
   legacy thing — manufacturers of microfiche, pager, dial-up,
   typesetting, classic-beer-style products — and file marks on
   vocabulary that reflects what they actually make. The vocabulary
   recedes because the rest of the field moves on, not because the
   firm is being clever. The persistence of these tokens at $50$–$80\%$
   survival in the corpus is consistent with this.

3. **Selection on durability.** Firms that survive long enough to
   file legacy-vocabulary trademarks have already proven they can
   stay in business. They were never going to fail at the same rate
   as a representative filing. The harvest-side lift is then a
   selection artefact: durable firms file legacy marks, durable
   firms also keep their marks alive.

The cluster-viability mediator (Section 5 of `paper/main.pdf`)
absorbs $84\%$ of the *quadratic* on $\Delta KL$ in the survival
regression, leaving a residual *linear* lift on the innovation side
($z = +2.65$). It does not separate the three mechanisms above. We
read the mediation result as: **flux extremes pay because they sit
in vocabulary clusters where peers also survive**, and the
flux-neutral middle sits in dying niches. That works for both sides
of $\Delta KL$ — both harvest tokens and innovation tokens are in
clusters where peers can support real businesses. But "live peers"
is itself the symptom, not the cause; what makes the clusters
viable is what we cannot disentangle.

## What is *not* in the harvest finding

This is not optimal distinctiveness symmetric. The Deephouse 1999
prediction is an inverted-$U$ on the *level* of distinctiveness:
too-similar marks lack identity, too-distinctive marks lack
legitimacy. Our $\Delta KL$ is not a level measure — it is a
directional measure of where a filing sits between the past and the
future of its category. The innovation-side response curve is
consistent with the empirical content of optimal distinctiveness on
the *positive* side (saturating returns, no inverted-$U$ turn-down).
The harvest-side response curve is *not predicted by* the optimal
distinctiveness literature. It is an empirical finding that, to our
knowledge, is novel.

It is also not the same as the kill-zone or killer-acquisition
mechanism (Kamepalli, Rajan & Zingales, 2022; Cunningham, Ederer &
Ma, 2021). Those operate on entry-suppression after a dominant firm
emerges. The harvest-side lift is observed on the *filings that
remain*, not on missing filings. We tested the kill-zone hypothesis
directly on a $2{,}536$-firm IPO event panel (see
[`paper/dynamism.pdf`](paper/dynamism.pdf) on the `business-dynamism`
branch) and found no chilling effect at the NICE-class level.

## Robustness checks that have been run

* **Per-(industry × year) classification** (Table 4 of
  `paper/main.pdf`). $32$ of $44$ NICE classes have a significant
  $+U$ on $\Delta KL$; only $1$ has a significant inverted-$U$
  (Telecommunications $038$); $11$ are indeterminate. The harvest
  lift is broad, not concentrated in a few classes.
* **Within-firm test** (falsification \#6 of `paper/main.pdf`). The
  $U$ on filing-level survival is preserved within firms that
  abandoned some trademarks but kept others alive (the "mixed"
  cohort, $349{,}819$ firms). The signal is filing-level, not
  firm-level.
* **Foreign-extension artefact** (falsification \#1). US-only and
  foreign-only sub-samples both produce the $U$ with nearly
  identical quadratic coefficients ($+0.0078$ and $+0.0100$). Not a
  Madrid Protocol artefact.
* **Token-level survival.** Hand-picked negative-stratum tokens
  (`microfiche`, `dial-up`, `pager`, `store-and-forward`, `lager`,
  `stout`, `shandy`, `annuity`, `microcomputers`) all survive at
  $50$–$80\%$ in the corpus. Flux-neutral tokens
  (`organisational`, `conformity`, `freighting`) survive at
  $11$–$26\%$.
* **Category-killer confound** (Section 4 of `paper/main.pdf`).
  Considered: vocabulary could recede because a dominant firm killed
  the category (Xerox/xerographic). Implausible at scale —
  $2.6$M debut filings across $44$ industries are too broad to be
  driven by category-killer monopolisation, none of the harvest-tail
  tokens we see are mid-20th-century genericisation episodes, and
  the token-level multi-filer evidence shows broad recedence.

## What we would want to do next

1. **Distinguish intelligent harvesting from passive
   late-occupation.** This requires a firm-side signal of intent
   that the trademark record does not provide. The natural extension
   is to merge with USPTO trademark *opposition* records — opposition
   filings reveal which marks the firm was prepared to defend, which
   is a behavioural signal of strategic commitment to the harvest
   niche.

2. **Firm-level mediation regression on financial outcomes.** Our
   firm-level financial result ($+3.2$ pp gross margin per
   $\sigma$ of trailing $\Delta KL$) has not yet been decomposed by
   sign. If only the positive side moves gross margin, the harvest
   finding is filings-survival-only. If the negative side also
   moves margin, the harvest is operationally substantive, not just
   a registration artefact.

3. **Token-vertical IPO chilling** (deferred from the dynamism
   paper). Replace the NICE-class operationalisation of "vertical"
   with cosine-similarity neighbourhoods of the IPO firm's
   goods/services prose. If a kill-zone effect exists in trademark
   filings, the token-vertical test is where it would show up.

4. **Reconcile the registration-vs-conditional-survival asymmetry.**
   The harvest-side lift sits mostly at the examiner gate; the
   innovation-side lift sits mostly post-registration. That
   asymmetry is itself unexplained and is the cleanest version of
   the "we don't understand the mechanism" claim.

## Bottom line

Negative $\Delta KL$ has a tighter, faster-saturating survival
response than positive $\Delta KL$ at observable magnitudes. The
harvest-tail finding is real, broad (32 of 44 industries),
token-corroborated, robust to the obvious confounds, and not
predicted by the literature we sit closest to. The mechanism is
not separable in the trademark record alone, and the most
informative test — distinguishing intelligent harvesting from
passive late-occupation — requires a behavioural signal we do not
yet have.

If you read only one thing from this corpus, read this finding.
