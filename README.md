# DrugRePurposeIR

DrugRePurposeIR searches biomedical literature for potential new uses of existing drugs.
Start with a disease, a drug, or a mechanism; get a ranked list of drugs or diseases with
supporting PubMed abstracts. The initial scope covers neurological and infectious diseases.

The project is written in Python. It includes data collection, indexing, several retrieval
methods, evaluation, and a small FastAPI backend. There is no frontend yet.

The IR implementations come from `reference/starter-code`: the tokenizers, inverted indexes,
lexical scorers, vector ranker, cross-encoder scorer, and MAP/nDCG functions. They are packaged
under [src/drug_repurpose_ir/ir](src/drug_repurpose_ir/ir), with their comments, docstrings, and
formatting preserved. Only imports between those files changed to package-relative imports.
The biomedical data adapters and paper-to-entity aggregation sit around that code.

## Setup

Use Python 3.12. The project runs on a Mac with CPU; a GPU is optional.

From the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[neural,api,dev]'
```

This is an editable install with the API and test tools. `pip install -e .` installs the core
dependencies, including the libraries imported by the reference modules. Model weights are
still loaded only when a neural method is used.

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

The default collection targets 500 PubMed results per disease plus an abacavir query, the full
openFDA Drugs@FDA export, B3DB classification data, and the archived repoDB CSV. The expanded
November 10 PubMed snapshot contains 9,588 unique abstracts; its shareable query/PMID/checksum
manifest is [here](data/manifests/pubmed-expanded-2026-11-10.json). Raw data, models, and results
are git-ignored, so a fresh clone needs its own downloads.

## Using the system

You can search from the terminal, [Python](#python-and-http-api), or the
[browser API page](#try-searches-in-the-browser).
Complete [Setup](#setup) first; a fresh clone does not include the collected data or indexes.

### Search from the terminal

With an existing dataset and index, start here:

```bash
source .venv/bin/activate
drugir doctor
drugir search "Alzheimer's" --method bm25 --k 5
```

`doctor` shows saved dataset/index counts and metadata. If no index exists yet but the raw files
are present, run `drugir run` first. BM25 needs only the lexical index. For dense or hybrid
search, build embeddings with `drugir index --dense`.

Other searches:

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

`results` is the ranked list. Read each result's `entity.name`, then inspect the supporting
paper titles, URLs, and `snippet` text under `evidence`. The top-level `warnings` list records
missing literature coverage and qualifications on BBB reranking.

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
| `bm25` | The supplied `Ranker` and `BM25` scorer over its inverted index. |
| `dense` | The supplied `VectorRanker.query()` and its dot-product scoring. |
| `hybrid` | BM25 and dense rankings combined with reciprocal rank fusion. |
| `cross-encoder` | Hybrid candidates rescored jointly with the query. Uses a general-domain MS MARCO model. |
| `abc` | Indirect associations through shared non-entity MeSH terms across papers. |
| `random` | Seeded ranking of eligible entities, including those without papers. |

The other supplied lexical scorers are also available: `tf-idf`, `dirichlet`, `pivoted`,
`word-count`, and `custom` (`YourRanker`). For example:

```bash
drugir search "Alzheimer's" --method tf-idf
drugir evaluate --methods bm25,tf-idf,dirichlet,pivoted,word-count,custom
```

Hybrid fusion adds `1 / (rrf_k + rank)` for each paper's position in each list. It combines
ranks because the underlying score scales differ. The supplied `CrossEncoderScorer.score()`
scores each paper in the resulting pool using the first 500 words of its raw text.

Entity aggregation supports `max`, `sum`, and `top-mean`. Scores are divided by the highest
positive entity score. If all scores are nonpositive, they are scaled between the minimum and
maximum instead, preserving their order. Negative BM25 scores are valid in the supplied formula
and are not discarded. Raw paper scores remain on the evidence records. With `max`, entities
sharing their best paper can tie.

For CNS disease pairs, BBB reranking adds `bbb_weight` for a positive label and subtracts it
for a negative label. Missing or conflicting labels receive no adjustment. This works in both
entity directions. Free-text searches without a disease context get no BBB adjustment.

## Code layout

All application code is in `src/drug_repurpose_ir/`. Start with `cli.py` to follow a command,
`models.py` for the data structures, or `engine.py` for the retrieval logic.

| File | Responsibility |
|---|---|
| [ir/](src/drug_repurpose_ir/ir) | Packaged reference algorithms; original code and documentation. |
| [reference_adapter.py](src/drug_repurpose_ir/reference_adapter.py) | Input JSONL, integer ID mapping, model setup, and metric-list padding. |
| [cli.py](src/drug_repurpose_ir/cli.py) | Commands and pipeline orchestration. |
| [models.py](src/drug_repurpose_ir/models.py) | Validated records used on disk and in responses. |
| [config.py](src/drug_repurpose_ir/config.py) | YAML settings, defaults, and path resolution. |
| [acquisition.py](src/drug_repurpose_ir/acquisition.py) | Downloads, retries, checksums, and reuse of local files. |
| [ingestion.py](src/drug_repurpose_ir/ingestion.py) | Source adapters, catalogue joins, deduplication, and coverage report. |
| [pubmed.py](src/drug_repurpose_ir/pubmed.py) | PubMed XML and publication dates. |
| [entities.py](src/drug_repurpose_ir/entities.py) | Canonical IDs, aliases, and mention matching. |
| [indexing.py](src/drug_repurpose_ir/indexing.py) | Build the reference index, save embeddings, check compatibility. |
| [engine.py](src/drug_repurpose_ir/engine.py) | Paper retrieval, entity scoring, BBB rule, and evidence selection. |
| [evaluation.py](src/drug_repurpose_ir/evaluation.py) | Query sets, metrics, bootstrap intervals, and run outputs. |
| [io.py](src/drug_repurpose_ir/io.py) | JSON/JSONL files and checksums. |
| [api.py](src/drug_repurpose_ir/api.py) | HTTP access to the same search engine. |

`tests/` contains small, fictional examples. `examples/demo/` supplies the runnable demo.
`reference/` contains the original files. Runtime imports use the packaged copies in `ir/`.
`ir/provenance.json` records source hashes and the few import substitutions. Tests compare the
copies byte-for-byte after those substitutions; formatting tools exclude this directory.

## Adding data

Add source files under `data/raw/`, then rebuild. All commands below run from the project root
with the virtual environment activated. These paths use `configs/default.yaml`; another config
may set a different `paths.raw`.

### 1. Choose the right folder and filename

For papers collected manually, use `data/raw/local/`. Create the folder if needed:

```bash
mkdir -p data/raw/local
```

You can keep separate files for each collection; there is no need to append to someone else's file.

| Content | Example path | Format |
|---|---|---|
| Papers or abstracts | `data/raw/local/lit-review.documents.jsonl` | One document record per line; schema below. `.documents.jsonl.gz` also works. |
| New drugs/diseases or metadata | `data/raw/local/lit-review.entities.jsonl` | One entity record per line. |
| Additional names for existing entities | `data/raw/local/lit-review.aliases.csv` | CSV with `entity_id,alias` columns. |
| Reviewed evaluation labels | `data/raw/local/lit-review.judgments.jsonl` | One drug/disease judgment per line. |

**The filename ending matters.** `lit-review.documents.jsonl` is accepted; `lit-review.jsonl`
and `notes.jsonl` are ignored. Plain `documents.jsonl`, `entities.jsonl`, `aliases.csv`, and
`judgments.jsonl` also work. Subfolders under `local/` are scanned too.

JSONL means one complete JSON object on each line, saved as UTF-8. Use double quotes, JSON
`true`/`false`/`null`, and no comments or trailing commas. Do not wrap the records in a JSON
array or spread one record over several lines. Unknown fields are rejected.

PDF, Word, Excel, plain-text files, and generic article CSVs are not imported directly.
Extract their article text and convert it to the document schema below. The reference code's
Wikipedia `docid`/`text` format is an internal indexing format; local uploads use `id`/`abstract`.

### 2. Add the papers

Save records in `data/raw/local/lit-review.documents.jsonl`. This is a fictional formatting
example; replace it with the actual paper and metadata before importing:

```jsonl
{"id":"local:team-study-001","title":"Example study title","abstract":"Example Drug and Alzheimer disease are mentioned in this fictional formatting example.","publication_date":"2021-06-30","date_precision":"month","entity_ids":["drug:example-drug","disease:alzheimer"],"source":"format-example","url":"https://example.org/papers/team-study-001"}
```

| Field | Required? | What to put here |
|---|---|---|
| `id` | Yes | A unique, stable string, such as `local:team-study-001`. Reuse the same ID when correcting that paper. |
| `abstract` | Yes | Nonempty evidence text from the source. Escape line breaks inside the string as `\n`. |
| `title` | No | The paper title; defaults to an empty string. |
| `publication_date` | No | `YYYY-MM-DD`, or `null` if unknown. |
| `date_precision` | No | `day`, `month`, `year`, or `unknown`; defaults to `unknown`. |
| `entity_ids` | No | List of existing drug/disease IDs. Omit it or use `[]` to rely on name matching. |
| `source`, `url`, `doi` | No | Source name, paper URL, and DOI. Include them when available so results can be traced back. |
| `mesh`, `chemicals` | No | Lists of MeSH terms and chemical names, such as `["Alzheimer Disease"]`. |
| `mesh_ids`, `chemical_ids` | No | Name-to-ID objects, such as `{"Alzheimer Disease":"D000544"}`. |

For a month-only date, use the month's last day and `date_precision: "month"`. For a year-only
date, use December 31 and `"year"`. Leave unknown dates as `null`; those papers are excluded
when a time cutoff is enabled.

The system finds known names/aliases in the title, abstract, MeSH terms, and chemical names.
Explicit `entity_ids` supplement those matches and must refer to entities in the catalogue.
They identify mentions, not proof that a drug treats a disease. Ordinary disease/drug retrieval
needs both the query entity and candidate entity linked to a supporting paper.

### 3. Add missing entities and aliases

Check `data/processed/entities.jsonl` for existing IDs, names, and aliases first. Reuse an
existing ID for the same entity. For a new entity, use a stable ID such as `drug:example-drug`
or `disease:example-disease` and add it to `data/raw/local/lit-review.entities.jsonl`:

```jsonl
{"id":"drug:example-drug","kind":"drug","name":"Example Drug","aliases":["Example compound X"],"approved":null,"bbb":null,"sources":["format-example"]}
```

Only `id`, `kind`, and `name` are required. `kind` must be `drug` or `disease`. Optional fields are:

| Field | Values |
|---|---|
| `aliases` | List of verified alternative names. |
| `identifiers` | Namespace-to-list object, such as `{"pubchem":["12345"]}`. |
| `cns` | `true` for a CNS disease that should use BBB reranking; otherwise `false`. |
| `approved`, `approved_since` | `true`/`false`/`null`, and an approval date in `YYYY-MM-DD` or `null`. |
| `bbb` | `"positive"`, `"negative"`, `"conflicting"`, or `null`. |
| `sources` | List of source references for the entity and any supplied metadata. |

Use `null` for unknown approval/BBB information. The default `approved_only: true` setting
excludes drugs without confirmed approval from search results, even if their papers were loaded.
Supply approval metadata only when supported by a source; openFDA can also supply it during import.

The document example above refers to `drug:example-drug`, so that entity record must be present
in the same import. `disease:alzheimer` already exists in the default config. Entities load before
documents, regardless of their filenames. For a new disease to join automatic PubMed acquisition
and the disease coverage report, also append its entity record under `diseases` in the config.

To add a name to an existing entity without writing another entity record, use
`data/raw/local/lit-review.aliases.csv`:

```csv
entity_id,alias
drug:example-drug,Example compound Y
```

These are also fictional names. Use reviewed synonyms for real records. Alias matching does not
infer equivalent salts or brands, and aliases shared by multiple IDs remain ambiguous. Local
entity records load before the external catalogues; put an alias there if it must reconcile a
source name during catalogue import. The alias CSV loads later and does not merge existing IDs
or transfer their approval/BBB metadata.

### 4. Add evaluation labels, if available

Papers can be searched without adding judgments. Judgments are separate, reviewed labels used
to score retrieval; they do not become searchable evidence. Put them in
`data/raw/local/lit-review.judgments.jsonl`:

```jsonl
{"drug_id":"drug:example-drug","disease_id":"disease:alzheimer","relevance":1,"source":"synthetic-format-example-only","status":"example"}
```

This is a fictional label, not a claim about a treatment. For real labels, `drug_id`,
`disease_id`, `relevance`, and `source` are required. Both IDs must exist with the corresponding
entity types. `relevance` is `1` for a reviewed positive or `0` for a reviewed negative;
`source` should identify the supporting dataset or evidence. `status` is optional.
Leave unreviewed pairs out. Queries need at least one positive judgment to enter evaluation.

Temporal evaluations also need verified `confirmed_date`, `prior_indication_date`, and
`prior_indication`; see [data-contracts.md](docs/data-contracts.md) for those rules.

### 5. Import, rebuild, and check

Run preparation first so schema/ID errors are caught before building embeddings:

```bash
drugir prepare
```

Fix any reported errors in the raw files and rerun. Check `data/processed/ingestion-report.json`:
`inputs` should include the new file, document counts should make sense, and `counts` and
`ambiguous_aliases` show duplicates, missing dates, and linking issues. The prepared
`documents.jsonl` contains the final `entity_ids` for each paper.

Then build and try a search:

```bash
drugir index --dense
drugir doctor
drugir search "Alzheimer's" --method hybrid --k 5
```

Use the disease or drug you added for the search. Check the returned evidence IDs/titles;
a successfully imported paper may still rank outside the candidate pool or be filtered by
approval, dates, or missing entity links. `drugir index` without `--dense` is enough for BM25,
but dense/hybrid searches need embeddings rebuilt for the changed corpus.

For later imports, `drugir run --dense` combines preparation and indexing. Add `--download`
only when you also want to fetch external datasets. Corpus changes currently rebuild the full
dense matrix. Restart a running API after rebuilding, and rerun evaluation after adding labels:

```bash
drugir evaluate --methods bm25,hybrid
```

Edit or remove records in the raw files, then repeat the import. Duplicate paper IDs, DOIs, or
normalized title/abstract text keep the first encountered record; adding another copy does not
update the original. Do not edit `data/processed/` or `artifacts/index/` directly. Files are not
watched in the background. Raw data is git-ignored, so share collected files separately when
another teammate needs the same corpus.

### Importing dataset exports

For the supported public datasets, keep the original export structure:

| Folder | Accepted files | Expected schema |
|---|---|---|
| `data/raw/pubmed/` | `.xml`, `.xml.gz`, `.jsonl`, `.jsonl.gz` | PubMed efetch XML, or the document JSONL schema above. |
| `data/raw/openfda/` | `.json`, `.json.zip` | **Drugs@FDA application exports**, with products/active ingredients and submissions; other openFDA endpoints use different schemas. |
| `data/raw/b3db/` | `.tsv`, `.tsv.gz` | B3DB classification columns `compound_name` and `BBB+/BBB-` (`BBB+` or `BBB-`); optional `CID`. |
| `data/raw/repodb/` | `.csv`, `.tsv` | Columns `drug_name`, `drug_id`, `ind_name`, `status`; optional `ind_id`. Statuses: Approved, Terminated, Suspended, Withdrawn. |

New PubMed XML exports can go straight into `data/raw/pubmed/`; the adapter extracts abstracts
and metadata. Records without abstracts and retracted records are skipped. For another data
source, convert it to the local schemas above or add a source adapter in `ingestion.py`.
The full contracts are in [data-contracts.md](docs/data-contracts.md).

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
| `acquisition.pubmed_per_disease` | `500` | PMID limit per disease or extra query. The initial November 8 baseline used 100. |
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

`retrieval.index_type` selects `basic` or `positional`, using the supplied index classes and
their save/load methods. The original BM25 defaults are `k1=1.2`, `b=0.75`, and `k3=8`.
`token_regex`, `stopwords`, and `minimum_word_frequency` are passed to the reference tokenizer
and Indexer. The adapter exports `artifacts/index/corpus.jsonl` with integer `docid` and `text`
fields; `lexical/index.json` is the reference index's own saved format. PubMed/local IDs remain
in `documents.jsonl` at the corresponding row positions.

Indexes record hashes of prepared data, the packaged reference code, and relevant settings.
If these change, rebuild before searching. Changes to the surrounding input/embedding adapters
still need a manual rebuild. Restart the API after rebuilding its index.

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
- **MAP@k:** calls the supplied `map_score()` unchanged, then averages query scores.
- **MRR@k:** reciprocal rank of the first positive, or zero if none appears within k.
- **nDCG@k:** calls the supplied `ndcg_score()` with judged gains and an ideal ordering.

The original MAP denominator is `min(positives in the supplied ranking, k)`, not the total
number of positive judgments. We retain that definition: a query can have MAP=1 while Recall
is below 1. For nDCG, short actual/ideal lists are padded with zeros because the supplied
function indexes exactly k entries. The scoring functions themselves are unchanged.

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

### Current baseline

The November 9, 2026 run used 1,922 abstracts and 127 positive pairs. There were 12 disease
queries and 125 drug queries with positive judgments.

| Method | Disease → drug Recall@10 | Drug → disease Recall@10 |
|---|---:|---:|
| BM25 | 0.2702 | 0.5640 |
| Dense | 0.2978 | 0.5320 |
| Hybrid | 0.2984 | 0.5720 |
| Hybrid + BBB | 0.2804 | 0.5720 |

Hybrid gave a small improvement over BM25; BBB reranking did not improve recall. Reverse-direction random
recall was already 0.524 because it returns ten of only twenty diseases. Only 66 positive pairs
had eligible approval metadata and 72 had direct literature co-occurrence, so missing data and
name matching account for much of the difficulty.

All eleven variants, MAP/nDCG results, and integration checks are in
[reference-integration.md](docs/reference-integration.md). The cross-encoder has only a two-query
smoke run with this backend and is not included here. The earlier library-based results remain
in [v1-validation.md](docs/v1-validation.md).

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

### Try searches in the browser

After setup and indexing, start the backend and leave that terminal running:

```bash
drugir serve --host 127.0.0.1 --port 8000
```

If you installed only the core dependencies, first run `pip install -e '.[api]'`.
Open [localhost:8000/docs](http://127.0.0.1:8000/docs). This page lets you run requests
and inspect the JSON responses directly:

1. Expand **GET /health**, click **Try it out**, then **Execute** to check the loaded document count.
2. Expand **GET /entities** and click **Try it out**. Set `kind` to `drug` and `q` to `abacavir`,
   then **Execute**. The response gives the matching catalogue names and IDs. Use `kind: disease`
   to look up diseases. This lookup searches preferred names, not aliases.
3. Expand **POST /search**, click **Try it out**, and replace the request body with:

```json
{
  "query": "abacavir",
  "direction": "drug-to-disease",
  "method": "bm25",
  "k": 5,
  "feasibility": false
}
```

Click **Execute** and read `results`, `evidence`, and `warnings` in the response. To search
for drugs instead, set `query` to `Alzheimer's` and `direction` to `disease-to-drug`.
Stop the server with Ctrl+C. Restart it after changing the data/index or configuration.

The same API works from another program or terminal:

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
models. `requirements-lock.txt` records the current Mac package versions; `pyproject.toml`
defines the dependencies for installation on other machines.

To run the supplied index/tokenizer/scorer/metric tests against the packaged code:

```bash
HF_HUB_OFFLINE=1 python scripts/check_reference.py
```

This needs the original test files under `reference/starter-code/tests`; fixtures are copied
to a temporary directory. Model-downloading cases are omitted. Project integration tests also
check that the pipeline uses the original classes and functions and preserves the source bytes.
The copied L2R/network modules are available with `pip install -e '.[learning]'`; they are not
enabled in the current pipeline without the corresponding training features or graph.

For new data sources, change `acquisition.py` and `ingestion.py`. For retrieval work, start in
`engine.py` and `indexing.py`; a new method also needs adding to `SearchRequest.method` and the
evaluation method list. Metric changes belong in `evaluation.py` and should have a small,
hand-calculated test. Keep experiment settings and query sets fixed when comparing methods.

## Common problems

| Problem | What to check |
|---|---|
| `drugir` is not found | Activate `.venv` or use `.venv/bin/drugir`. |
| Browser says the site cannot be reached | Run `.venv/bin/drugir serve` from the project root and keep it running, then open `http://127.0.0.1:8000/docs`. The link alone does not start the server. |
| No documents or index | Fetch/import data, then run `drugir run`. |
| Stale index | Rebuild with `drugir run --dense` for the same config. |
| Missing dense index | Run `drugir index --dense` to download the model and build embeddings. |
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
