# DrugRePurposeIR

Search biomedical literature for drug repurposing candidates. Enter a disease or drug and get
ranked results with supporting PubMed abstracts. The initial scope is neurological and
infectious diseases.

The Python backend uses our assignment code from `reference/starter-code` for indexing,
scoring, and MAP/nDCG. Packaged copies are in [ir/](src/drug_repurpose_ir/ir), with the original
comments and docstrings preserved.

## Setup

Use Python 3.12. Run these commands from the project root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e '.[api,dev]'
drugir run --download --dense
```

This downloads PubMed, openFDA Drugs@FDA, B3DB, repoDB, and the embedding model, then builds
the indexes. A Mac CPU works; a GPU is optional. Data and models are git-ignored, so a fresh
clone needs its own downloads. See [.env.example](.env.example) for optional NCBI credentials.

If raw data is already present, use `drugir run --dense` without `--download`.
For a small offline example, follow the [demo instructions](docs/usage.md#try-the-demo).

## Search

```bash
# Disease → drugs
drugir search "Alzheimer's" --method hybrid --k 5

# Drug → diseases
drugir search abacavir --direction drug-to-disease --method hybrid --k 5
```

Use `--method bm25` for lexical search or add `--feasibility` for BBB reranking.
Search prints JSON; `--output runs/search.json` saves it. Read `entity.name` and the supporting
`evidence` titles, URLs, and excerpts. Scores are relative to the query, not probabilities.

To try searches in a browser:

```bash
drugir serve
```

Keep that terminal running and open [localhost:8000/docs](http://127.0.0.1:8000/docs).
Expand **POST /search** and click **Try it out**. See the
[browser walkthrough and Python examples](docs/usage.md) for request bodies and other query types.

## Add data

Use these folders and filename endings:

| Data | Location | Format |
|---|---|---|
| PubMed | `data/raw/pubmed/` | Efetch `.xml` / `.xml.gz` |
| openFDA | `data/raw/openfda/` | Drugs@FDA `.json` / `.json.zip` |
| B3DB | `data/raw/b3db/` | Classification `.tsv` / `.tsv.gz` |
| repoDB | `data/raw/repodb/` | Original archive `.csv` / `.tsv` |
| Collected papers | `data/raw/local/team.documents.jsonl` | JSONL; requires `id`, `abstract` |
| Drugs/diseases | `data/raw/local/team.entities.jsonl` | JSONL; requires `id`, `kind`, `name` |
| Extra names | `data/raw/local/team.aliases.csv` | CSV columns `entity_id,alias` |
| Evaluation labels | `data/raw/local/team.judgments.jsonl` | JSONL; requires `drug_id`, `disease_id`, `relevance`, `source` |

JSONL means one UTF-8 JSON object per line. For example, replace this with an actual paper:

```jsonl
{"id":"local:study-001","title":"Study title","abstract":"Text from the source abstract."}
```

The filename ending matters: `team.documents.jsonl` is recognized; `notes.jsonl` is ignored.
IDs must stay consistent across files. PDFs and arbitrary CSVs need conversion first.
Full schemas, examples, and proposal-source requirements are in the [data import guide](docs/adding-data.md).

After adding or editing raw files:

```bash
drugir prepare
drugir index --dense
drugir doctor
```

Check `data/processed/ingestion-report.json` for the new inputs, duplicates, and linking issues.
Edit raw files rather than generated files, and restart the API after rebuilding. Drugs need
confirmed approval metadata to appear under the default settings.

The current source plan uses openFDA instead of DrugBank. Full DrugBank, RxNorm, and MONDO
releases require conversion to local records; structured targets/mechanisms are not loaded yet.

## Evaluate

```bash
drugir evaluate --methods bm25,hybrid,hybrid+feasibility
```

Reports go under `runs/`. The current benchmark measures known indication recovery.
See [metrics and evaluation rules](docs/methodology.md) and [current results](docs/reference-integration.md#evaluation-results).

## Development

Edit [configs/default.yaml](configs/default.yaml) for diseases, corpus size, model/device settings,
and the time cutoff. Start in [cli.py](src/drug_repurpose_ir/cli.py) to follow a command or
[engine.py](src/drug_repurpose_ir/engine.py) for retrieval.

```bash
pytest -q
ruff check src tests scripts
ruff format --check src tests scripts
```

## Documentation

| Guide | Contents |
|---|---|
| [Adding data](docs/adding-data.md) | Source formats, folders, examples, and rebuild steps |
| [Data contracts](docs/data-contracts.md) | Record fields and identifier mappings |
| [Using the system](docs/usage.md) | Terminal, browser, and Python examples |
| [Code and configuration](docs/codebase.md) | Pipeline, file map, settings, and troubleshooting |
| [Evaluation](docs/methodology.md) | Metrics, output files, and time splits |
| [Reference integration](docs/reference-integration.md) | Reused code, verification, and results |
| [Sources](docs/sources.md) | Dataset versions, attribution, and licenses |
| [Roadmap](docs/roadmap.md) | Remaining work |
