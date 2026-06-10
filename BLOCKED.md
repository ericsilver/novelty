# Remaining construct-validity tests — what's done, what's blocked

State as of 2026-05-26. The `data/` and `reference_data/` trees are empty on this
branch, so anything needing the raw ~10M-filing corpus or external reference data
is blocked. The firm-year panels in `data_publish/` are present, so the firm-level
tests ran. Decision rules are stated so the interpretive call is made ex ante.

## Done now (firm-year panels only)
| Test | Script | Result |
|---|---|---|
| Within-firm ∆KL × patent timing | `scripts/wsC_within_firm_patents.py` | ∆KL is a within-firm temporal **substitute** for patents (−0.05σ/log-patent, t≈−8) |
| Convergent/discriminant validity | `scripts/wsD_validity_battery.py` | ∆KL ⟂ patents (Spearman −0.015) but **+0.3–0.5σ higher** for BCG/MIT innovators |
| Returns instability (+28pp) | `scripts/wsE_returns_diagnostic.py` | Non-monotone, collapses at 5y → **drop** debut figure; keep firm-year |
| Paper | `paper/construct_validity_note.tex` → `.pdf` | Compiled, 2 figures |

## Blocked on the raw corpus (`make tm` rebuilds it — ~12 GB download, multi-hour)
Each test below is either fully scripted or precisely specified; all consume
`data/processed/{tm_class<NNN>,surprise_class<NNN>}.parquet`.

- **Interaction estimand (§0).** `scripts/blocked/interaction_estimand.py` — COMPLETE,
  runs as-is once the corpus exists. Builds the **kernel-independent** churn measure
  (vocabulary turnover + filing-volume growth — never touches KL) that the critique
  required, then fits `dkl ~ logpat * churn` with firm+year FE.
  *Rule:* interaction>0 & main effect≈0 ⇒ ∆KL's innovation content lives only in
  churning classes (a firm×class-phase quantity, consistent with diffusion-as-innovation).

- **Term provenance (three-way).** Reuse `src/novelty/token_attribution.py` machinery.
  Split each filing's ∆KL into corpus-new / class-new(recombination) / class-extant
  tokens; re-estimate survival & returns on each component.
  *Rule:* effect loads on recombination ⇒ rename the firm component "recombinant
  novelty," not foresight. Needs per-filing tokens + a whole-corpus first-appearance index.

- **Half-life sensitivity (F0c).** Re-run `recompute_h2.py` at H∈{1,2,4}y (prod=2y),
  record the **sign-flip rate** of ∆KL. *Rule:* >20–30% flips ⇒ report ∆KL as kernel-relative.

- **ID-Manual codification (Workstream E) — highest external value.** Do high-∆KL
  filings use terms the USPTO ID Manual *later* standardizes, while those terms are
  still rare in-class? This is the strongest available external test of the diffusion
  claim. Needs the ID Manual **version history** (USPTO publishes periodic IDML updates).

- **Attorney/correspondent FE (Workstream A).** Re-parse USPTO XML for correspondent
  of record (incremental to the country re-parse). *Rule:* effects survive within-attorney
  ⇒ drafting fashion alone insufficient. NOT a clean innovation extractor (counsel sorting).

## Blocked on external linkage
- **External survival outcome (Workstream V).** Link a stratified validation subsample
  to bankruptcy (PACER) or active-web-presence to break the §8-maintenance circularity
  in the survival U-shape. Only a few-thousand-firm sample is needed, not full relink.

- **BISG demographics (Workstream G).** Replace surname-only imputation in the
  ethnic-cluster work. Restore the surname file with
  `git checkout -- reference_data/census_surnames/Names_2010Census.csv`, then add
  block-group geography. **First verify** `owner_address` is the owner's address, not
  the correspondent's — if it's the agent's, the geographic prior is poisoned.

## Note on a memory discrepancy
A saved memory says "∆KL = abs diff." The code and published data use the **signed**
∆KL = prospective − retrospective KL (= DeDeo resonance); `firm_year_patents_and_dkl.csv`
confirms `mean_dkl = mean_pros − mean_retr`. If an `abs`-diff variant exists on another
branch, it is a different statistic and these results do not transfer to it.
