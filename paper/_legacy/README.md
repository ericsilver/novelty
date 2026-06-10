# Legacy papers

These papers represent earlier framings of the research line, preserved here for reference. The current paper that supersedes most of them is `paper/ssrn_diffusion_paper.pdf` at the repository root.

| File | Pages | Status |
|---|---|---|
| `main.{tex,pdf}` | 223 | The original long-form paper framing ΔKL as an innovation measure with a U-shape in 5-year mark survival. The U-shape finding is preserved (now reported in `ssrn_diffusion_paper.pdf` §4 at both the filing and token levels); the strong-sense "innovation measure" framing is qualified by the within-firm patent-orthogonality result, which is also in the new paper. The halflife chart in this PDF (`paper/results/halflife_signal_class009.png`) has a known data-load issue (duplicate H=2.0 row, ambiguous flat-5y placement at x=5); regenerate from `scripts/halflife_signal.py` before reuse. |
| `short.{tex,pdf}` | 7 | Short version of `main.tex`. |
| `dynamism.{tex,pdf}` | 13 | A separate line: USPTO debut coverage of US business formation, plus three failed crowding tests. |
| `dynamism_review.{tex,pdf}` | 8 | Brief review of the dynamism work. |
| `ethnic_clusters_note.{tex,pdf}` | 21 | Another separate line: US-only ethnic clusters in trademark filings; patent guild signal collapse once foreign inventors are stripped; Black trademark-rise vs patent-fall divergence. |
| `construct_validity_note.{tex,pdf}` | — | A working note that became Part I of `integrated_report.tex`. Material is now in `ssrn_diffusion_paper.pdf` §2 and §6. |
| `diffusion_phase0_note.{tex,pdf}` | — | A working note that became Part II of `integrated_report.tex`. Material is now in `ssrn_diffusion_paper.pdf` §§3–7. |
| `integrated_report.{tex,pdf}` | 17 | Consolidated working report combining the two notes above. Superseded by the focused SSRN paper. |

The data and analyses underlying these papers remain in `data_publish/`, `scripts/`, and `paper/results/`. The papers themselves are kept here as a historical record of how the research line developed.
