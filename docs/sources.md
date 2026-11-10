# Sources and reproducibility

The first acquisition on November 8, 2026 uses the following sources. Per-file manifests under
`data/raw/` record download URLs, checksums, timestamps, and public query parameters. Models and
source data retain their own licenses; this repository does not relicense them.

## Relation to the proposal

The Data section of [Project Proposal.pdf](<../Project Proposal.pdf>) specifies PubMed,
DrugBank or openFDA, B3DB, repoDB, and RxNorm/DrugBank plus MeSH/MONDO identifiers. The current
implementation uses the proposal's openFDA alternative. Its native loaders accept PubMed efetch
XML, Drugs@FDA JSON/ZIP, B3DB classification TSV, and the original archived repoDB CSV/TSV.

DrugBank/MeSH identifiers are retained from repoDB/PubMed. Additional reviewed RxNorm, DrugBank,
MeSH, or MONDO mappings can be supplied as local entity JSONL. Raw DrugBank, RxNorm, and MONDO
releases have no direct parser yet. Drugs@FDA supplies approval evidence rather than DrugBank's
structured targets/mechanisms. The proposed Hetionet/Open Targets graph extensions are future work.
See the [dataset import instructions](adding-data.md#importing-the-proposal-datasets) for the
accepted files, folders, fields, and conversion requirements.

## Sources used in the initial run

| Source | Initial selection | Primary reference |
|---|---|---|
| PubMed | 100 ranked results per configured disease, 1990–2026; abstracts only, drug-therapy/repurposing query; extra abacavir query | [E-utilities reference](https://www.ncbi.nlm.nih.gov/books/NBK25499/) |
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

The FDA adapter uses original approved application submissions, not mere NDC listing, as its
approval signal. It deliberately skips combination products. Missing catalogue matches and BBB
joins are recorded; exact-name joining is an initial baseline, not a complete chemical crosswalk.
