# Data sources, provenance, and terms

Every external dataset this project reads, where it came from, and what may be
done with it. The repository's own licence covers the code and the derived
panels in `data_publish/`; it does not and cannot cover the third-party sources
below, which retain their own terms.

## Primary sources

**USPTO Trademark Annual Applications backfile (TRTYRAP)**
US Government work, public domain (17 U.S.C. §105). Retrieved through the USPTO
Open Data Portal API; a free API key is required and is read from `.env` (never
committed). Approximately 12 GB, streamed once by `make tm`. The raw backfile is
not redistributed here — `data/` is gitignored — because it is large and
available at source.

**SEC EDGAR Financial Statement Data Sets, and the SEC company-ticker file**
US Government work, public domain. Retrieved by `scripts/download_sec_fsds.py`
from `sec.gov`. SEC's access policy requires a descriptive User-Agent carrying a
contact address; set `SEC_USER_AGENT` in `.env` rather than using the author's.
Fair-access rate limits apply and the download script respects them.

**PatentsView**
Licensed **CC-BY-4.0**. Retrieved by `scripts/fetch_patentsview.py`. Attribution
is required of anyone redistributing derived data: cite PatentsView as the
source of the patent assignee and grant-date tables. The derived firm-year panel
in `data_publish/firm_year_patents_and_dkl.csv` contains PatentsView-derived
counts and inherits that obligation.

**SEC Form D quarterly data sets**
US Government work, public domain. Used for the Regulation D financing rows.

## Comparator sources used in validation only

**BCG "Most Innovative Companies" (2021–2023) and MIT Technology Review "50
Smartest Companies" (2015)**
Company names hand-transcribed from published rankings into
`data_publish/comparators/`. Facts are not copyrightable and only the company
names are reproduced, but the lists are the publishers' editorial work and are
used here solely to test whether the measure agrees with expert judgement. The
paper reports that it does not, on a sample too small to carry weight.

**Crunchbase Open Data Map (historical mirror)**
**Not redistributed in this repository.** The comparator analysis in
`scripts/external_compare.py` reads a Crunchbase company export obtained from the
`notpeter/crunchbase-data` historical mirror of the ~2013–2015 Open Data Map,
which is licensed **CC-BY-NC 4.0**. The non-commercial clause is incompatible
with this repository's code licence, and the attribution requirement is not
satisfied by a mirrored copy, so the raw file is excluded. To reproduce that one
analysis, fetch the export yourself and place it at
`data_publish/comparators/crunchbase_companies.csv`. Nothing in the paper depends
on it.

## Derived data published here

`data_publish/` contains panels computed by this project from the sources above:
firm-year mean scores, the SEC-matched panel, and the PatentsView-joined panel.
These are original derived work, offered under **CC-BY-4.0**, except that rows
derived from PatentsView carry PatentsView's own CC-BY-4.0 attribution
requirement, which is satisfied by citing both.

No source in this project contains personal data about private individuals
beyond what USPTO and SEC publish as a matter of public record: trademark owner
names and SEC registrant names, which are published by those agencies.
