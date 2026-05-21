# novelty

**KL surprise scores for U.S. trademark goods/services descriptions across the
full USPTO backfile (~16.5M filings, 1985–2025), joined to SEC EDGAR
financials and PatentsView patent counts.** We measure each filing's
*vocabulary flux* — how much its goods/services prose has moved with or
against where its NICE-class language was heading — and ask how flux
predicts post-registration survival, financial outcomes, public-listing
rates, and how the corpus aligns with the declining-business-dynamism
literature.

Two papers live in this repo:

| PDF | What it argues |
|---|---|
| [`paper/main.pdf`](paper/main.pdf) | $\Delta KL$ as an innovation measure — comparable magnitude to patents, independent of patents, and with a survival response curve that *aligns* with optimal-distinctiveness on the innovation side and *diverges* on the harvest side with a separate (and much tighter) response curve |
| [`paper/short.pdf`](paper/short.pdf) | The same finding in 7 pages with the headline figures and the falsification table |
| [`paper/dynamism.pdf`](paper/dynamism.pdf) (on `business-dynamism` branch) | What fraction of US business formation the trademark corpus captures, the surname-based ethnic composition of debut filers, and **three failed tests of the crowding hypothesis** as a possible reading of declining business dynamism |

Two deep-dive READMEs:

* [`README_HARVEST.md`](README_HARVEST.md) — the surprisingly strong evidence
  that *negative* $\Delta KL$ also pays, with unclear cause. Read this if
  you want to know the most interesting thing in the corpus.
* [`PROPOSAL.md`](PROPOSAL.md) — original research design

## Clear findings

### From the $\Delta KL$ paper (`paper/main.pdf`)

1. **$\Delta KL = KL_\text{pros} - KL_\text{retr}$ is an innovation measure
   of comparable magnitude to patent counts and uncorrelated with them.**
   Per-firm-year $r = +0.010$ on $174{,}569$ firm-years. On 4-year stock
   excess returns over the S\&P 500, $\Delta KL$ alone has incremental
   $R^2 = +0.0046$ over controls; patents alone have incremental
   $R^2 = +0.0024$; jointly $+0.0069$, essentially additive — two
   independent innovation channels.

2. **The innovation tail aligns with optimal distinctiveness.** On
   debut filings, $P(\text{survived 5y})$ on the positive-$\Delta KL$
   side fits a saturating exponential
   $y = 0.995 - 0.55\,e^{-0.063\,\Delta KL}$ with $R^2 = 0.499$:
   monotone-increasing in prospective novelty, concave-down,
   diminishing returns, no inverted-$U$ inflection. This is the
   empirical content of optimal distinctiveness (Deephouse, 1999;
   Zhao, Fisher, Lounsbury & Miller, 2017) translated from *level of
   distinctiveness* to *directional vocabulary flux*.

3. **The harvest tail is the surprise: a separate saturating
   exponential, much tighter fit, with unclear mechanism.** On the
   negative-$\Delta KL$ side, the same functional form fits at
   $R^2 = 0.978$: $y = 0.573 - 0.137\,e^{-3.33\,|\Delta KL|}$.
   Survival rises faster with $|\Delta KL|$ and saturates closer to
   zero. The deeper a debut filing sits into the receding legacy
   vocabulary of its category, the more durable its mark. Mechanism
   is not separable in the trademark record (intelligent harvesting
   vs.\ passive late-occupation vs.\ selection on an unobservable).
   See [`README_HARVEST.md`](README_HARVEST.md).

4. **The survival $U$ on $\Delta KL$ holds at the per-industry level
   in 32 of 44 NICE classes.** Only $1$ industry has a significant
   inverted-$U$ (Telecommunications $038$); the remaining $11$ are
   indeterminate (no significant year-level curvature in either
   direction), not inverted.

5. **A cluster-viability mediator absorbs 84% of the quadratic on
   survival.** Defining the "vertical" of a filing as its
   cosine-similarity neighbours on goods/services TF-IDF in a
   $[t-2y, t+5y]$ window, $\log(1 + \text{live\_competitors})$
   absorbs $84\%$ of the $\Delta KL^2$ coefficient on $5y$ survival
   while leaving the linear $\Delta KL$ effect intact ($z = +2.65$).
   The $U$ is a niche-viability signal; the residual linear lift on
   the innovation side is what cluster viability alone cannot
   reproduce.

6. **Nine alternative explanations ruled out.** Foreign-extension
   artefact (no), first-time vs.\ recurring filer artefact (no),
   firm-level vs.\ filing-level artefact (no — the $U$ survives
   within mixed-cohort firms), large-incumbent harvest-tail
   survivor bias (partial), "novel product in old vocabulary"
   confound (no), crypto/NFT-wave imitated-but-failed counterexample
   (no), vintage pooling artefact (no — the $U$ holds in $23/32$
   per-year cohorts), Google Books English-language-drift confound
   (partial, $\sim 22\%$ shared variance per token; not redundant).

### From the business-dynamism paper (`paper/dynamism.pdf`, `business-dynamism` branch)

7. **USPTO trademark debut counts track Census BFS business
   applications** at $r = +0.94$ on levels and $r = +0.56$ year-over-year,
   2005–2025. The trademark corpus is a defensible higher-resolution
   complement to BFS at NICE-class resolution.

8. **First-time-filer share is *rising*, not falling.** Pooled debut
   share trough $46$--$49\%$ in 1996–2003, current 2020–2025 level
   $56$--$58\%$. The conventional aggregate-crowding reading of declining
   business dynamism is not visible in trademark filings. The rise is
   composition-driven (consumer-services classes growing at the expense
   of tech-info classes).

9. **Top-10 owner concentration is falling in every NICE class.** HHI
   in Telecommunications fell from 48.4 to 4.4; Sci~\& Tech from 3.3 to
   0.5; Software from 3.2 to 0.4. The FAANG-crowding hypothesis is not
   supported anywhere in the corpus.

10. **No IPO chilling effect at the NICE-class level.** Event study on
    $877$ class-events from $2{,}536$ IPO firms ($2012$–$2022$), with
    class and year fixed effects, finds a post-vs-pre coefficient on
    $\log(1 + \text{debut count})$ of $+0.007$ ($t = +0.67$,
    $p = 0.50$). The Kamepalli/Rajan/Zingales (2022) kill-zone effect
    is not visible at this granularity. Token-vertical refinement
    (cosine-similarity neighbourhoods of the IPO firm's pre-IPO
    goods/services prose) is flagged as future work — the
    class-level null does not rule out a finer-grained chilling
    effect.

11. **Surname-based ethnic composition shifts dramatically toward
    API-coded names.** Applying the Word/Coleman/Nunziata/Kominski
    (2008) Census surname/race-ethnicity probability file to the
    $1.83$M individual-name debut filers in our corpus: $5\%$
    API-coded in the 1990s rising to $33\%$ in 2020–2024. We flag
    the well-documented 2018–2021 foreign-applicant surge from
    China as a confound and discuss what an unconfounded estimate
    would require (applicant country from the USPTO XML, not
    currently in our processed panel).

## What's in this repo

| Path | What it holds |
|---|---|
| `src/novelty/` | Library code: USPTO XML parser, CountVectorizer dictionary, decay-weighted KL surprise (`surprise_decay.py` is the H=2y production reference), per-filing→firm-year rollup, survival outcomes |
| `scripts/` | One CLI script per analysis or paper figure; the Makefile wires them into the build graph |
| `data/raw/` | SEC EDGAR FSDS quarterly ZIPs + PatentsView dumps |
| `data/processed/` | Per-NICE-class Parquet panels: `tm_class<NNN>`, `vocab_class<NNN>`, `surprise_class<NNN>`, `firm_year_class<NNN>`, `outcomes_class<NNN>` + SEC firm-year financials + the USPTO↔SEC crosswalk |
| `reference_data/` | Public reference datasets checked into the repo: Census BFS, Census 2010 surnames, Census BDS, NICE/NAICS crosswalk |
| `paper/main.tex` | Long-form $\Delta KL$ paper |
| `paper/short.tex` | Seven-page short version |
| `paper/dynamism.tex` | Business-dynamism paper (on `business-dynamism` branch) |
| `paper/results/` | Every figure and table the papers `\input` or `\includegraphics`; regenerated by `make analysis` |
| `Makefile` | The one-button reproducible build: `make all` walks setup → download → analyse → compile |
| `Makefile.dynamism` | Pipeline for the business-dynamism paper (on `business-dynamism` branch) |
| `PROPOSAL.md` | Original research design |
| `README_HARVEST.md` | Deep-dive on the negative-$\Delta KL$ finding |

## How to build the papers

```bash
# end-to-end (requires Python 3.11 and ~20GB free for the TM backfile)
make setup tm sec crosswalk analysis paper

# or simply
make all

# for the business-dynamism paper
git checkout business-dynamism
make -f Makefile.dynamism download analysis review
```

`make help` lists the individual targets. The build is incremental: editing
one analysis script and re-running `make paper` rebuilds only that figure
and re-compiles the LaTeX. `make clean-results` forces a full figure
rebuild.

The TM backfile download is heavy (`make tm` takes hours and writes
~15 GB to `data/processed/`). Once the per-class Parquet panels and the
SEC crosswalk exist, `make analysis paper` regenerates everything from
them in well under an hour and produces `paper/main.pdf`.

Requires Python 3.11, a TeX install (TeX Live), and a free USPTO Open Data
Portal API key from <https://data.uspto.gov> (My ODP → My API Key); set it
via `cp .env.example .env` and edit `USPTO_ODP_API_KEY=...`.

## Stage timing on a fresh checkout

| Stage | Wall-clock | Output |
|---|---|---|
| `setup` | < 5 min | `.venv/` |
| `tm` | hours | `data/processed/{tm,vocab,surprise,firm_year,outcomes}_class<NNN>.parquet` × 45 |
| `sec` | minutes | `data/processed/sec_firm_year.parquet` |
| `crosswalk` | minutes | `data/processed/uspto_sec_crosswalk.parquet` |
| `analysis` | ~ 1 hour | `paper/results/*.{png,csv,tex,txt,json}` |
| `paper` | < 1 min | `paper/main.pdf` and `paper/short.pdf` |

## Adding a new analysis

Drop a script into `scripts/`, write its output into `paper/results/`, add
a target for it in the Makefile (one line for the script invocation, plus
an entry in the `FIGURES` list so `make paper` notices it), and `\input` /
`\includegraphics` the result from `paper/main.tex`.

## Data not in this repo

The raw USPTO backfile (~12 GB), per-class records/vocab/surprise/outcome
Parquets (~15 GB), and SEC EDGAR FSDS ZIPs (~7 GB) are excluded by
`.gitignore`. The reproduction pipeline regenerates them from public
sources.

## References — primary literature cited in these papers

* Akcigit, U., & Ates, S. T. (2021). Ten facts on declining business
  dynamism and lessons from endogenous growth theory. *American
  Economic Journal: Macroeconomics*, 13(1), 257–298.
* Barron, A. T., Huang, J., Spang, R. L., & DeDeo, S. (2018).
  Individuals, institutions, and innovation in the debates of the
  French Revolution. *PNAS*, 115(18), 4607–4612.
* Cunningham, C., Ederer, F., & Ma, S. (2021). Killer acquisitions.
  *Journal of Political Economy*, 129(3), 649–702.
* Decker, R. A., Haltiwanger, J., Jarmin, R. S., & Miranda, J.
  (2014). The role of entrepreneurship in US job creation and
  economic dynamism. *Journal of Economic Perspectives*, 28(3),
  3–24.
* Deephouse, D. L. (1999). To be different, or to be the same? It's
  a question (and theory) of strategic balance. *Strategic
  Management Journal*, 20(2), 147–166.
* Dinlersoz, E., Goldschlag, N., Myers, A., & Zolas, N. (2018). An
  anatomy of US firms seeking trademark registration. CES Working
  Paper 18-46.
* Hall, B. H., Jaffe, A., & Trajtenberg, M. (2005). Market value and
  patent citations. *RAND Journal of Economics*, 36(1), 16–38.
* Kamepalli, S. K., Rajan, R., & Zingales, L. (2022). Kill zone.
  *AEJ: Microeconomics*, forthcoming.
* Kogan, L., Papanikolaou, D., Seru, A., & Stoffman, N. (2017).
  Technological innovation, resource allocation, and growth.
  *Quarterly Journal of Economics*, 132(2), 665–712.
* Murdock, J., Allen, C., & DeDeo, S. (2017). Exploration and
  exploitation of Victorian science in Darwin's reading notebooks.
  *Cognition*, 159, 117–126.
* Word, D. L., Coleman, C. D., Nunziata, R., & Kominski, R. (2008).
  Demographic aspects of surnames from Census 2000. US Census Bureau
  working paper.
* Zhao, E. Y., Fisher, G., Lounsbury, M., & Miller, D. (2017).
  Optimal distinctiveness: Broadening the interface between
  institutional theory and strategic management. *Strategic
  Management Journal*, 38(1), 93–113.
