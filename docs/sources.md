# Sources and reproducibility

The first acquisition on November 8, 2026 uses the following sources. Per-file manifests under
`data/raw/` record download URLs, checksums, timestamps, and public query parameters. Models and
source data retain their own licenses; this repository does not relicense them.

| Source | Initial selection | Primary reference |
|---|---|---|
| PubMed | November 8 baseline: 1,922 unique abstracts from up to 100 ranked results per configured disease plus the abacavir query. Expanded November 10 snapshot: 9,588 unique abstracts from 500-result query caps, using the same 1990–2026 filters. | [E-utilities reference](https://www.ncbi.nlm.nih.gov/books/NBK25499/) |
| openFDA | Full Drugs@FDA bulk export, November 6, 2026, 29,446 application records | [Drugs@FDA](https://open.fda.gov/apis/drug/drugsfda/), [bulk download metadata](https://api.fda.gov/download.json) |
| B3DB | Classification TSV pinned to commit `75dab1cc607bb7a03f3de5c576cffd223e767844` | [Author repository](https://github.com/theochem/B3DB), [dataset paper](https://pubmed.ncbi.nlm.nih.gov/34716354/) |
| repoDB | Original archived final database, Figshare article 3811674, CSV file 7341422 | [Author repository](https://github.com/adam-sam-brown/repoDB), [paper](https://doi.org/10.1038/sdata.2017.29), [archive](https://api.figshare.com/v2/articles/3811674) |
| Dense encoder | `NeuML/pubmedbert-base-embeddings`; resolved model commit recorded in `dense.json` | [Model card](https://huggingface.co/NeuML/pubmedbert-base-embeddings) |
| Cross-encoder | `cross-encoder/ms-marco-MiniLM-L6-v2`, general-domain baseline | [Model card](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2) |

B3DB is released under CC0; cite the authors' Scientific Data paper. The original repoDB data
are available under CC BY 4.0; its usage notes request citation of repoDB, AACT, and DrugBank.
Using this published archive does not require access to the separate DrugBank database. The v1
archive is intentionally reproducible and older than the current repoDB web application.

PubMed abstracts can be copyrighted by their publishers. Keep downloaded corpora local unless
their applicable terms allow redistribution, and follow the [NCBI disclaimer](https://www.ncbi.nlm.nih.gov/About/disclaimer.html).
The downloader limits requests to below three per second and supports an optional NCBI API key
and contact email, following the [E-utilities usage guidance](https://eutilities.github.io/site/API_Key/usageandkey/).

The 100-per-disease baseline remains the reference snapshot reported in
[`v1-validation.md`](v1-validation.md). The expanded acquisition selected up to 500 PMIDs per
query: 10,500 query results were selected, 10,307 records were parsed, and 719 duplicate records
were removed, leaving 9,588 unique abstracts. The committed
[`PubMed snapshot manifest`](../data/manifests/pubmed-expanded-2026-11-10.json) preserves the
queries, selected PMIDs, per-file checksums, and coverage counts without redistributing abstract
text. Raw XML and normalized corpus files remain local and are not committed. Rebuild the local
snapshot with `drugir fetch --source pubmed` followed by `drugir prepare`, then refresh the
shareable identifier manifest with `python scripts/export_pubmed_manifest.py`.

The FDA adapter uses original approved application submissions, not mere NDC listing, as its
approval signal. It deliberately skips combination products. Missing catalogue matches and BBB
joins are recorded; exact-name joining is an initial baseline, not a complete chemical crosswalk.
