# Public data artifacts

These two CSVs are the public-facing data contribution of the
companion paper. Both are produced by the pipeline in `../scripts/`
and are subset to the publicly-traded firms that the SEC EDGAR
Financial Statement Data Sets cover, so that other researchers can
reproduce or extend the firm-level findings without re-running the
full 16.5 M-filing trademark sweep or the 9.4 M-patent PatentsView
extract.

## `firm_year_dkl.csv`  (~ 4.5 MB, 59,033 rows)

One row per (firm × calendar year) for every USPTO-trademark owner
that we successfully linked to a SEC EDGAR filer (CIK). Annual
trademark-novelty scores aggregated across all 45 NICE classes a
firm filed in.

| column | meaning |
|---|---|
| `cik` | SEC Central Index Key |
| `sec_name` | Company name in SEC FSDS / ticker file |
| `owner_name` | USPTO-recorded owner-of-filing string |
| `year` | Calendar year of filing |
| `n_filings` | Filings the firm made that year (clean, in 1995--2020) |
| `mean_pros` | Mean prospective KL across the year's filings |
| `mean_retr` | Mean retrospective KL across the year's filings |
| `mean_dkl` | Mean ΔKL = pros − retr (innovator score) |

## `firm_year_patents_and_dkl.csv`  (~ 22 MB, 174,569 rows)

One row per (firm × calendar year) for every USPTO-trademark owner
whose normalized name also matches a PatentsView assignee. Includes
both the trademark dKL and the firm's patent count, suitable for
testing convergent validity between the two innovation measures.

| column | meaning |
|---|---|
| `normalized_name` | Owner name after suffix-stripping normalization |
| `uspto_owner_name` | USPTO-recorded owner-of-filing string |
| `patentsview_name` | PatentsView disambiguated assignee organization |
| `year` | Calendar year |
| `mean_pros`, `mean_retr`, `mean_dkl` | Trademark novelty scores |
| `n_filings` | USPTO trademark filings that year |
| `n_patents` | USPTO patents granted to this assignee that year |

## Reproduction

The full pipeline (USPTO download → per-class vocabulary → KL surprise →
firm aggregation → SEC join → PatentsView join) is in `../scripts/`.
See the top-level `README.md` and `Makefile` for invocation.
