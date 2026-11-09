# DrugRePurposeIR

DrugRePurposeIR searches biomedical literature for potential new uses of existing drugs.
Start with a disease, a drug, or a mechanism; get a ranked list of drugs or diseases with
supporting PubMed abstracts. The initial scope covers neurological and infectious diseases.

The project is written in Python. It includes data collection, indexing, several retrieval
methods, evaluation, and a small FastAPI backend. There is no frontend yet.

## Setup

Use Python 3.12. The project runs on a Mac with CPU; a GPU is optional.

From the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[neural,api,dev]'
```

This is an editable install with the dependencies for neural retrieval, the API, and tests.
For BM25 and the non-neural baselines only, `pip install -e .` is enough.

Optional NCBI credentials can go in `.env`; see [.env.example](.env.example). Downloads work
without a key. Model weights are cached under `artifacts/models/` unless `HF_HOME` is set.

### Try the demo

The demo uses fictional data and separate output folders. Once dependencies are installed,
these commands need no network access:

```bash
drugir run --config configs/demo.yaml
drugir search "Brain disease" --config configs/demo.yaml
drugir evaluate --config configs/demo.yaml --methods random,bm25,bm25+feasibility
```

### Download the data and build the indexes

```bash
drugir run --download --dense
drugir doctor
```

The first command downloads the configured datasets, prepares the records, and builds the
lexical and dense indexes. The first neural run also downloads pretrained model weights.
`doctor` prints the current dataset counts and index metadata.

The default collection is 100 PubMed results per disease plus an abacavir query, the full
openFDA Drugs@FDA export, B3DB classification data, and the archived repoDB CSV. Raw data,
models, and results are git-ignored, so a fresh clone needs its own downloads.

## Search

```bash
# Disease → drugs
drugir search "Alzheimer's" --method hybrid --k 5

# Drug → diseases
drugir search abacavir --direction drug-to-disease --method hybrid

# Free-text mechanism query
drugir search "reverse transcriptase inhibition" --direction mechanism --method hybrid

# Apply the BBB reranker
drugir search "Alzheimer's" --method hybrid --feasibility --k 5
```

Search prints JSON. Add `--output runs/search.json` to save it. Use `drugir search --help`
for the full options. If the virtual environment is not activated, call `.venv/bin/drugir`.

Entity queries must match a known name, alias, or ID. For example, `disease:alzheimer` works
as well as the disease name. Use `--direction mechanism` for free text.

Each result includes:

- The drug or disease, its identifiers, and available approval/BBB metadata.
- `retrieval_score`: the entity score before BBB reranking.
- `feasibility_adjustment`: the BBB contribution.
- `score`: the final ranking score.
- `evidence_count`: matching papers in the retrieved pool.
- `evidence`: up to three paper references and abstract excerpts by default.

Scores are relative to the current query, not probabilities. A BBB bonus can push a score
above one. The excerpts are selected from the abstracts; they are not generated explanations.

## How it works

```text
PubMed abstracts ──→ clean and deduplicate ──→ link drug/disease mentions ──→ indexes
                                                     ↑
                               disease aliases + openFDA + B3DB

query ──→ retrieve papers ──→ aggregate scores by entity ──→ optional BBB reranking

repoDB + curated labels ──→ evaluate the resulting entity rankings
```

1. **Collect and prepare data.** Source adapters convert the raw files into common entity,
   document, and judgment records. Duplicate papers are removed by ID, DOI, or normalized text.
2. **Link names.** A dictionary matcher finds drug and disease aliases in abstracts and their
   MeSH/chemical metadata. Ambiguous aliases remain unresolved. Salt forms are not merged
   automatically.
3. **Index papers.** BM25 uses title, abstract, and MeSH tokens. The dense index contains
   normalized title/abstract embeddings in the same order as the document file.
4. **Retrieve and aggregate.** Direct methods retrieve papers, then rank entities mentioned
   alongside the query entity. The default entity score comes from its highest-scoring paper.
5. **Rerank and evaluate.** The optional BBB rule adjusts CNS pairs. Evaluation uses separate
   judgments; repoDB outcome labels never become searchable document text.

### Retrieval methods

| Method | Implementation |
|---|---|
| `bm25` | Lexical matching with `rank_bm25.BM25Okapi`. |
| `dense` | PubMedBERT embeddings with exact cosine search. |
| `hybrid` | BM25 and dense rankings combined with reciprocal rank fusion. |
| `cross-encoder` | Hybrid candidates rescored jointly with the query. Uses a general-domain MS MARCO model. |
| `abc` | Indirect associations through shared non-entity MeSH terms across papers. |
| `random` | Seeded ranking of eligible entities, including those without papers. |

Hybrid fusion adds `1 / (rrf_k + rank)` for each paper's position in each list. It combines
ranks because BM25 and cosine scores have different scales. The cross-encoder scores only
the resulting candidate pool.

Entity aggregation supports `max`, `sum`, and `top-mean`. After aggregation, scores are divided
by the highest positive entity score for that query. With `max`, entities sharing their best
paper can tie.

For CNS disease pairs, BBB reranking adds `bbb_weight` for a positive label and subtracts it
for a negative label. Missing or conflicting labels receive no adjustment. This works in both
entity directions. Free-text searches without a disease context get no BBB adjustment.

## Code layout

All application code is in `src/drug_repurpose_ir/`. Start with `cli.py` to follow a command,
`models.py` for the data structures, or `engine.py` for the retrieval logic.

| File | Responsibility |
|---|---|
| [cli.py](src/drug_repurpose_ir/cli.py) | Commands and pipeline orchestration. |
| [models.py](src/drug_repurpose_ir/models.py) | Validated records used on disk and in responses. |
| [config.py](src/drug_repurpose_ir/config.py) | YAML settings, defaults, and path resolution. |
| [acquisition.py](src/drug_repurpose_ir/acquisition.py) | Downloads, retries, checksums, and reuse of local files. |
| [ingestion.py](src/drug_repurpose_ir/ingestion.py) | Source adapters, catalogue joins, deduplication, and coverage report. |
| [pubmed.py](src/drug_repurpose_ir/pubmed.py) | PubMed XML and publication dates. |
| [entities.py](src/drug_repurpose_ir/entities.py) | Canonical IDs, aliases, and mention matching. |
| [indexing.py](src/drug_repurpose_ir/indexing.py) | Saved tokens, document embeddings, and index compatibility checks. |
| [engine.py](src/drug_repurpose_ir/engine.py) | Paper retrieval, entity scoring, BBB rule, and evidence selection. |
| [evaluation.py](src/drug_repurpose_ir/evaluation.py) | Query sets, metrics, bootstrap intervals, and run outputs. |
| [io.py](src/drug_repurpose_ir/io.py) | JSON/JSONL files and checksums. |
| [api.py](src/drug_repurpose_ir/api.py) | HTTP access to the same search engine. |

`tests/` contains small, fictional examples. `examples/demo/` supplies the runnable demo.
`reference/` contains earlier assignments and is not imported by the project.

## Data

Put new files in the appropriate raw-data folder, then rebuild:

```bash
drugir run --dense
```

This processes local files. Add `--download` when you also want to acquire data. Files are not
watched in the background, and arbitrary files in the project root are not picked up.

| Folder | Accepted inputs | Purpose |
|---|---|---|
| `data/raw/pubmed/` | PubMed `.xml[.gz]` or normalized `.jsonl[.gz]` | Literature. |
| `data/raw/openfda/` | Drugs@FDA `.json` or `.json.zip` exports | Approval catalogue. |
| `data/raw/b3db/` | Classification `.tsv[.gz]` | BBB metadata. |
| `data/raw/repodb/` | Original-schema `.csv` or `.tsv` | Vocabulary and evaluation labels. |
| `data/raw/local/` | `*entities.jsonl`, `*documents.jsonl[.gz]`, `*judgments.jsonl`, `*aliases.csv` | Curated additions. |

For local files, the suffix matters: `team.documents.jsonl` is recognized; `notes.jsonl` is not.
JSONL stores one JSON object per line. Required fields and examples are in
[data-contracts.md](docs/data-contracts.md).

Preparation writes `entities.jsonl`, `documents.jsonl`, `judgments.jsonl`, and
`ingestion-report.json` under `data/processed/`. Check the report for missing joins, duplicate
papers, ambiguous aliases, and disease coverage. Edit raw inputs rather than these generated files.

Local entities are loaded before the external catalogues. If a reviewed alias should make two
source names resolve to one ID during import, include it in the local entity record. An alias
added afterward through `aliases.csv` does not merge existing IDs or move their metadata.

## Configuration

Edit [configs/default.yaml](configs/default.yaml), or copy it for an experiment:

```bash
cp configs/default.yaml configs/experiment.yaml
drugir run --config configs/experiment.yaml --dense
drugir evaluate --config configs/experiment.yaml --methods bm25,hybrid
```

Keep experiment configs directly under `configs/`: relative data paths resolve from the config's
grandparent directory. Use distinct output paths if experiments should keep separate working indexes.

The settings most likely to change are:

| Setting | Default | Controls |
|---|---|---|
| `diseases` | 20 starter diseases | Scope, aliases, and CNS status. |
| `acquisition.pubmed_per_disease` | `100` | PMID limit per disease or extra query. |
| `acquisition.pubmed_start_year` / `pubmed_end_year` | `1990` / `2026` | Downloaded publication range. |
| `acquisition.pubmed_queries` | Abacavir query | Additional literature searches. |
| `retrieval.candidate_documents` | `200` | Retrieved paper pool; separate from final entity `--k`. |
| `retrieval.aggregation` | `max` | How supporting papers contribute to an entity. |
| `retrieval.approved_only` | `true` | Exclude candidate drugs without approval evidence. |
| `retrieval.bbb_weight` | `0.15` | BBB score adjustment. |
| `retrieval.device` | `cpu` | `cpu`, `mps`, or `cuda`. |
| `evaluation.ks` | `[5, 10, 20]` | Metric cutoffs. |
| `evaluation.cutoff_year` | `null` | Optional literature/approval date bound. |

The remaining model, batching, scoring, and bootstrap defaults are documented in
[config.py](src/drug_repurpose_ir/config.py). Changing the collection scope requires fetching
the new papers before rebuilding; configuration alone cannot supply missing literature.

Indexes record hashes of prepared data and relevant settings. If these change, rebuild the
index before searching. Code changes to tokenization or encoding are not fingerprinted, so use
a fresh index path when changing those algorithms. Restart the API after rebuilding its index.

For Apple Silicon acceleration, try `device: mps` with a compatible PyTorch installation.
On Great Lakes or another CUDA server, use `device: cuda` and create an environment on that
machine; do not copy the Mac virtual environment. Scheduler settings depend on your allocation.
Reduce `batch_size` if memory runs out.

## Evaluation

```bash
# Quick check on the first four queries
drugir evaluate --methods random,bm25,abc --max-queries 4

# Full comparison using the prepared dense index
drugir evaluate --methods bm25,dense,hybrid,hybrid+feasibility

# Evaluate one direction or save a named run
drugir evaluate --methods bm25,hybrid --directions disease-to-drug
drugir evaluate --methods bm25,hybrid --output runs/experiment-01
```

Search enables BBB adjustment with `--feasibility`; evaluation uses the `+feasibility` suffix.
`--max-queries` takes a stable prefix, not a random sample. Choose a new output directory for
each named run; completed summaries are not overwritten.

The default benchmark is **known indication recovery**. RepoDB approved pairs are positives;
terminated, suspended, and withdrawn pairs are benchmark negatives. Conflicting outcomes are
excluded. Unjudged pairs remain unjudged in the data and score zero in the ranking metrics.

- **Recall@k:** retrieved positives divided by all positives for a query, including those missing
  from the corpus or excluded by approval filtering.
- **MRR@k:** reciprocal rank of the first positive, or zero if none appears within k.
- **nDCG@k:** rewards placing positives earlier in the list.

The output also reports unjudged results, returned counts, BBB rates, coverage, bootstrap
intervals, and paired differences against BM25. Means weight queries equally. Keep the two
directions separate when interpreting results: their candidate sets and query counts differ.

Each run writes to `runs/`:

| File | Contents |
|---|---|
| `summary.json` | Metrics, intervals, exclusions, and model/data metadata. |
| `per-query.csv` | Individual query scores and coverage counts. |
| `queries.jsonl` | The query set and judgments used. |
| `rankings.jsonl` | Ranked entities and supporting papers for every variant. |
| `config.json` | Resolved settings for the run. |

`runs/LATEST` is a text file pointing to the latest evaluation directory.

### Initial results

The November 8, 2026 run used 1,922 abstracts and 127 positive pairs. There were 12 disease
queries and 125 drug queries with positive judgments.

| Method | Disease → drug Recall@10 | Drug → disease Recall@10 |
|---|---:|---:|
| BM25 | 0.2792 | 0.5640 |
| Dense | 0.2978 | 0.5320 |
| Hybrid | 0.3044 | 0.5720 |
| Hybrid + BBB | 0.2863 | 0.5720 |

Hybrid gave a small improvement; BBB reranking did not improve recall. Reverse-direction random
recall was already 0.524 because it returns ten of only twenty diseases. Only 66 positive pairs
had eligible approval metadata and 72 had direct literature co-occurrence, so missing data and
name matching account for much of the difficulty.

The full results and caveats are in [v1-validation.md](docs/v1-validation.md). The cross-encoder
has only a four-query evaluation so far and is not included in this table.

### Time splits

`cutoff_year` filters papers and drug availability by date. For a true repurposing test, set
`mode: temporal-repurposing` and supply reviewed labels with `confirmed_date`,
`prior_indication_date`, and a different `prior_indication`. The archived repoDB CSV alone lacks
these facts. Current BBB labels and pretrained models are not historical snapshots.
See [methodology.md](docs/methodology.md) for the full evaluation rules.

## Python and HTTP API

Use the engine directly from Python:

```python
from drug_repurpose_ir.config import load_config
from drug_repurpose_ir.engine import SearchEngine

engine = SearchEngine(load_config("configs/default.yaml"))
response = engine.search("Alzheimer's", method="bm25", k=5)
for result in response.results:
    print(result.entity.name, result.score)
    for paper in result.evidence:
        print("  ", paper.title, paper.url)
```

Reuse an engine across queries to reuse loaded models and caches. Create a new one after
changing its data or settings.

To start the backend:

```bash
drugir serve --host 127.0.0.1 --port 8000
```

Open [localhost:8000/docs](http://127.0.0.1:8000/docs) to try requests. The routes are
`GET /health`, `GET /entities`, and `POST /search`.

```bash
curl http://127.0.0.1:8000/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"Alzheimer disease","method":"bm25","k":5}'
```

## Development

```bash
pytest -q
ruff check src tests
ruff format --check src tests
```

Tests use fictional records and temporary directories. They do not download scientific data or
models. `requirements-lock.txt` records the initial Mac package versions; `pyproject.toml`
defines the dependencies for installation on other machines.

For new data sources, change `acquisition.py` and `ingestion.py`. For retrieval work, start in
`engine.py` and `indexing.py`; a new method also needs adding to `SearchRequest.method` and the
evaluation method list. Metric changes belong in `evaluation.py` and should have a small,
hand-calculated test. Keep experiment settings and query sets fixed when comparing methods.

## Common problems

| Problem | What to check |
|---|---|
| `drugir` is not found | Activate `.venv` or use `.venv/bin/drugir`. |
| No documents or index | Fetch/import data, then run `drugir run`. |
| Stale index | Rebuild with `drugir run --dense` for the same config. |
| Missing dense index | Install the neural extra and run `drugir index --dense`. |
| Unknown query | Check its ID/aliases, or use mechanism mode for free text. |
| Empty results | Check mentions, approval joins, and co-occurrence in the ingestion report. |
| Familiar drug is filtered | Its exact record may lack approval metadata because of a salt/name split. |
| Model fails offline | Download its weights first; offline mode needs a complete local cache. |

## Open issues

The main gaps are entity reconciliation, treatment-relation scoring, and literature coverage.
Abacavir and abacavir sulfate currently have separate records, splitting approval and BBB
metadata. Some Alzheimer's candidates come from symptom-management papers; abacavir can return
stroke because a paper measured cardiovascular events during HIV treatment. These are retrieval
errors to fix before treating the rankings as repurposing evidence.

[data-contracts.md](docs/data-contracts.md) covers file formats,
[sources.md](docs/sources.md) lists datasets and attribution, and
[roadmap.md](docs/roadmap.md) tracks the planned research work.
