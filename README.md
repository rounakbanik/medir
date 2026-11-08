# DrugRePurposeIR

DrugRePurposeIR is a Python research system for finding literature-supported connections between
existing drugs and diseases. It supports **disease → drugs**, **drug → diseases**, and free-text
**mechanism → entities** searches. Results are ranked entities with supporting papers, extracted
sentences, and separate retrieval/feasibility scores.

This repository contains a working **v1**: public-source acquisition, data adaptation, entity
normalization, lexical and neural retrieval, BBB feasibility reranking, evaluation, tests, and an
optional HTTP API. The assignment files under `reference/` are reference material only; the
application never imports them. The proposal PDF describes the original study scope.

**Research limitation:** appearing together in a relevant paper does not establish a therapeutic
relationship. V1 can retrieve symptom-management, adverse-event, background-chemical, or
combination-regimen mentions. Scores are rankings, not probabilities of clinical efficacy.
The initial examples and benchmark below explicitly expose these limitations.

## Contents

1. [Where a new teammate should start](#where-a-new-teammate-should-start)
2. [Installation and first successful run](#installation-and-first-successful-run)
3. [Repository map](#repository-map)
4. [How the pipeline works](#how-the-pipeline-works)
5. [Adding and understanding data](#adding-and-understanding-data)
6. [Retrieval and scoring](#retrieval-and-scoring)
7. [Configuration reference](#configuration-reference)
8. [Commands and everyday workflows](#commands-and-everyday-workflows)
9. [Reading search results](#reading-search-results)
10. [Evaluation and interpretation](#evaluation-and-interpretation)
11. [Python and HTTP interfaces](#python-and-http-interfaces)
12. [Reproducibility, caches, and hardware](#reproducibility-caches-and-hardware)
13. [Testing and contributing](#testing-and-contributing)
14. [Troubleshooting](#troubleshooting)
15. [Known limitations and next steps](#known-limitations-and-next-steps)

## Where a new teammate should start

To **run** the system, install dependencies and try the isolated synthetic demo below. Then search
the real corpus, inspect the papers behind a result, and run a small evaluation. You do not need
to understand every algorithm before doing that.

To **understand or modify** the implementation, read in this order:

| Order | File | What to learn |
|---|---|---|
| 1 | [models.py](src/drug_repurpose_ir/models.py) | Entity, document, judgment, request, result, and evidence contracts. |
| 2 | [default.yaml](configs/default.yaml) and [config.py](src/drug_repurpose_ir/config.py) | Configurable choices, defaults, validation, and path resolution. |
| 3 | [cli.py](src/drug_repurpose_ir/cli.py) | Public commands and the functions they orchestrate. |
| 4 | [ingestion.py](src/drug_repurpose_ir/ingestion.py) | How sources become canonical records; how labels stay separate from search text. |
| 5 | [entities.py](src/drug_repurpose_ir/entities.py) and [pubmed.py](src/drug_repurpose_ir/pubmed.py) | Alias linking, ambiguity, XML, and publication-date precision. |
| 6 | [indexing.py](src/drug_repurpose_ir/indexing.py) | Snapshot fingerprints, cutoff filtering, and document/vector row alignment. |
| 7 | [engine.py](src/drug_repurpose_ir/engine.py) | Paper retrieval, entity aggregation, BBB adjustment, and evidence snippets. |
| 8 | [evaluation.py](src/drug_repurpose_ir/evaluation.py) | Query construction, metrics, coverage, uncertainty, and experiment artifacts. |
| 9 | [acquisition.py](src/drug_repurpose_ir/acquisition.py), [io.py](src/drug_repurpose_ir/io.py), [api.py](src/drug_repurpose_ir/api.py) | Downloading, file handling, and the optional HTTP layer. |
| 10 | [tests/](tests/) | Small fictional cases specifying expected behavior without scientific downloads. |

Every module, class, function, and method has a docstring. Inline comments explain processing
steps, assumptions, scoring, and edge cases. Start with the record contracts before changing a
stage's input or output.

Two useful call chains to trace:

```text
drugir run
  → cli.run()
  → settings_from() / load_config()
  → acquisition.fetch()        [only with --download]
  → ingestion.ingest()
  → indexing.build_index()

drugir search
  → cli.search()
  → SearchEngine.__init__()    [load/validate the snapshot]
  → SearchEngine.search()
  → retrieve() / _retrieve()   [paper rankings for direct methods]
  → aggregate()               [paper scores become entity scores]
  → BBB adjustment and evidence
```

## Installation and first successful run

### Requirements

- Python **3.11–3.14** according to package metadata. Initial verification used **Python 3.12
  on Apple Silicon**, which is the recommended starting point for reproducing the local v1.
- Internet access for package installation, initial datasets, and initial neural-model downloads.
- Local storage for raw exports, normalized records, model weights, and evaluation runs.
- A GPU is optional. CPU is the default and supports the starter corpus.

Run the examples from the repository root containing `pyproject.toml`, `configs/`, and `src/`.

### Fresh environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[neural,api,dev]'
drugir --help
```

Editable installation means edits under `src/` are used without reinstalling. Restart an existing
Python process or API server after changing its implementation.

| Installation | Functionality |
|---|---|
| `python -m pip install -e .` | Acquisition, preparation, BM25, random/ABC, non-neural evaluation. |
| `python -m pip install -e '.[neural]'` | Adds SentenceTransformers/PyTorch for dense, hybrid, cross-encoder. |
| `python -m pip install -e '.[api]'` | Adds FastAPI/Uvicorn for the backend. |
| `python -m pip install -e '.[dev]'` | Adds pytest/Ruff; the optional API test skips without FastAPI. |
| `python -m pip install -e '.[neural,api,dev]'` | Full development environment used here. |

Without activating the environment, use `.venv/bin/drugir` and `.venv/bin/python` directly.

### Optional environment variables

Start from `.env.example`. If `.env` already exists, edit it; otherwise copy the example to `.env`.
Existing shell variables take precedence over values loaded from this file.

```dotenv
NCBI_EMAIL=
NCBI_API_KEY=
OPENFDA_API_KEY=
HF_HOME=artifacts/models
TOKENIZERS_PARALLELISM=false
```

NCBI contact/key values are included in requests when populated. The current FDA adapter uses
public bulk exports and does not consume `OPENFDA_API_KEY`; that entry is reserved for a future
authenticated adapter. Credentials are optional for the supplied public acquisition workflow.
Model-cache defaults keep weights under the project unless an existing `HF_HOME` overrides them.

### First run with fictional data, without scientific downloads

```bash
drugir run --config configs/demo.yaml
drugir search "Brain disease" --config configs/demo.yaml --method bm25
drugir evaluate --config configs/demo.yaml --methods random,bm25,bm25+feasibility
```

This requires installed base dependencies but no network after installation. The committed demo
is entirely **synthetic** and isolated from the real study:

| Artifact | Location |
|---|---|
| Inputs | `examples/demo/local/` |
| Prepared records | `data/demo/` |
| Index | `artifacts/demo/` |
| Runs | `runs/demo/` |

Fictional Alpha/Beta examples specify behavior, not scientific findings. Do not mix their labels
with the real benchmark.

### First run with real sources

```bash
# Acquire configured sources, normalize records, and build both index types.
drugir run --download --dense

# Inspect loaded data and manifests.
drugir doctor

# Try both directions.
drugir search "Alzheimer's" --method hybrid --k 5
drugir search abacavir --direction drug-to-disease --method bm25 --k 5

# Start evaluation with inexpensive methods.
drugir evaluate --methods random,bm25,bm25+feasibility,abc
```

Initial neural use can download weights. You do not need to download/rebuild before every search.
In an existing prepared checkout, try `doctor` and `search` first. Raw data, models, and runs are
git-ignored, so teammates cloning the code must acquire/import those artifacts separately.

## Repository map

```text
medir/
├── Project Proposal.pdf         Original study proposal
├── README.md                    Onboarding and operating guide
├── pyproject.toml               Package, dependency groups, CLI entry point, tool settings
├── requirements-lock.txt        Initial Mac environment's exact package versions
├── .env.example                 Optional environment template
├── configs/
│   ├── default.yaml             Real-data starter scope and study settings
│   └── demo.yaml                Isolated synthetic configuration
├── src/drug_repurpose_ir/
│   ├── models.py                Shared validated records
│   ├── config.py                YAML validation and path resolution
│   ├── acquisition.py           Cached/rate-limited downloads
│   ├── pubmed.py                PubMed XML adaptation and date parsing
│   ├── entities.py              Catalogue, aliases, and dictionary linking
│   ├── ingestion.py             Source joins, deduplication, and coverage reporting
│   ├── indexing.py              Snapshot files and aligned document embeddings
│   ├── engine.py                Retrieval, aggregation, feasibility, and evidence
│   ├── evaluation.py            Labels, metrics, uncertainty, and experiment artifacts
│   ├── io.py                    JSON/JSONL, individual-file replacement, checksums
│   ├── cli.py                   Terminal orchestration
│   └── api.py                   Optional HTTP wrapper around the same engine
├── tests/                       Fictional, offline regression tests
├── examples/demo/local/         Committed synthetic inputs
├── docs/
│   ├── data-contracts.md         Formats and schema examples
│   ├── methodology.md           Metric and temporal-study interpretation
│   ├── sources.md               Source/model references and attribution
│   ├── v1-validation.md         Initial verification and benchmark
│   └── roadmap.md               Next six weeks of research work
├── data/raw/                    Acquired/imported source inputs
├── data/processed/              Generated real-data JSONL and report
├── artifacts/index/             Generated current search snapshot
├── artifacts/models/            Default model cache
├── runs/                        Experiment/search outputs and LATEST pointer
└── reference/                   Prior assignments; never imported
```

The normal collaboration surfaces are `src/`, `configs/`, `tests/`, `examples/`, and `docs/`.
`data/`, `artifacts/`, and `runs/` are local/generated working artifacts. Add curated source records
to raw inputs and regenerate; do not patch a result by manually editing processed/index files.

## How the pipeline works

### Stage 1: acquire raw sources

| Source | Data | Purpose |
|---|---|---|
| PubMed | Titles, abstracts, MeSH/chemical names and IDs, DOI, dates | Searchable literature and entity mentions. |
| Drugs@FDA through openFDA | Application submissions, single-ingredient products, brands | Independent approval metadata and observed dates. |
| B3DB | Compound names, BBB classification, PubChem IDs | Matched feasibility metadata. |
| Archived repoDB | Drug/indication names and IDs, outcome statuses | Vocabulary plus independent benchmark judgments. |

For PubMed, each configured disease and extra anchor query is a separate search job. Search
returns bounded PMID samples; efetch retrieves their XML records in batches. The match count can
be much larger than the sample downloaded. The starter configuration samples 100 per job.

Each download has a neighboring `.manifest.json` with safe query parameters, source URL, checksum,
retrieval timestamp, and source context. Matching identity and checksum allow reuse; `--refresh`
bypasses it. Requests are paced and retry throttling/server/transport failures. Pacing is per
downloader instance, not coordinated across teammates sharing a network.

`pubmed/active-files.json` records the latest successful managed batch selection. Old managed
batches outside that selection are excluded from preparation. Manual files without download
manifests remain eligible. `LAST_FETCH.txt` lists dataset files made ready, including reused files.

### Stage 2: build and enrich entities

Source order matters inside `ingest()`:

1. Copy configured diseases and load local `*entities.jsonl` records.
2. Load repoDB drug vocabulary; label status does not confer independent FDA approval.
3. Add FDA single-ingredient approval observations and brand aliases.
4. Add local alias crosswalks to existing canonical records.
5. Match B3DB observations to catalogue drugs; retain conflicting BBB evidence.
6. Construct independent repoDB/local judgments.
7. Validate, deduplicate, and link literature.

An `Entity.id`, such as `disease:alzheimer`, is the stable local join key. Source identifiers remain
in their namespaces: MeSH, UMLS, DrugBank, FDA application, PubChem, and so forth. Published IDs
from archived data do not imply access to a separate licensed DrugBank database.

The matcher lowercases names and normalizes punctuation/accents, then finds unambiguous whole-token
aliases using Aho-Corasick. It is a dictionary matcher, not a trained biomedical NER model or
chemical-structure resolver. It does not infer salt equivalence or disease-parent relationships.
ASCII normalization can also lose non-ASCII-only symbols; review biomedical shorthand carefully.

An alias with multiple possible IDs remains ambiguous. Short names and ID-like strings are
restricted in free-text mention scanning; direct typed query resolution can still use canonical
or namespaced IDs.

### Stage 3: normalize and deduplicate literature

PubMed XML becomes `Document` records, preserving nested formatting text, structured abstract
labels, source IDs, DOI, and date precision. Missing PMID/abstract records and explicitly retracted
publication records are skipped. Local JSONL documents must already follow the common schema.

Duplicate IDs, normalized DOI values, or normalized title/abstract text are collapsed. Files are
sorted and the first duplicate encountered wins. This is deterministic but does not reconcile
conflicting versions; review source conflicts separately.

Mentions come from title/abstract text, MeSH names, chemical names, and valid explicit local
`entity_ids`. Mentioning an entity does not establish its role as a treatment.

Preparation writes these contracts under `paths.processed`:

| File | Contents |
|---|---|
| `entities.jsonl` | Canonical catalogue and provenance/approval/BBB metadata. |
| `documents.jsonl` | Deduplicated literature and linked IDs. |
| `judgments.jsonl` | Separate pair labels, source statuses, optional reviewed dates. |
| `ingestion-report.json` | Counts, input hashes, ambiguities, join/coverage diagnostics. |

Writers replace each completed file individually. Several outputs together are not a database
transaction. Do not run preparation/rebuilds concurrently against the same working directories.

### Stage 4: create a searchable snapshot

`build_index()` stores these files under `paths.index`:

- `documents.jsonl`: selected papers, optionally filtered by cutoff.
- `entities.jsonl`: catalogue metadata snapshot.
- `tokens.jsonl`: lexical tokens from title, abstract, and MeSH names.
- `manifest.json`: input/settings fingerprints, count, date exclusions.
- Optional `embeddings.npy` and `dense.json`: aligned numeric rows and model metadata.

Search reconstructs the BM25 scorer from saved tokens. V1 does not require an external search or
vector database. Dense retrieval uses an exact, memory-mapped NumPy matrix. Row `i` must represent
document `i` in the saved file; never reorder the matrix or document list independently.

A configured literature cutoff is applied before tokenization/encoding. Papers with unknown dates
are excluded. Month/year-only dates use the end of the known interval, with precision retained;
that does not assert an exact publication day.

### Stage 5: retrieve papers and rank entities

`SearchEngine` validates the snapshot, reconstructs BM25, and builds entity/document/MeSH lookup
maps. Neural weights and embedding rows load lazily.

For direct retrieval methods, a query flows through:

1. Resolve a drug/disease name, alias, or ID; mechanism mode accepts free text.
2. Expand known entities with aliases and retrieve a bounded paper pool.
3. Require query/candidate co-mentions in a selected paper.
4. Apply candidate eligibility, including approved-only drug filtering by default.
5. Aggregate paper scores into one score per candidate entity.
6. Normalize within the query and optionally apply the BBB pair rule.
7. Return up to `k` entities with strongest paper references and short extracted sentences.

Within one engine, query vectors and a bounded paper-ranking cache avoid repeated work. The
engine is not a directory watcher. Recreate it, or restart the API, after rebuilding or changing
the settings it represents.

### Stage 6: evaluate final entity rankings

Each usable pair supplies labels for both query directions. Evaluation constructs one fixed query
manifest and reuses it for every requested variant. It scores final entity IDs, not paper IDs.
Judgment statuses are not copied into searchable text or used as retrieval scores.

## Adding and understanding data

### Recognized input folders and formats

Put files in the supported source folders. Arbitrary files in the project root are not ingested
automatically, and adding a file does not trigger a background rebuild.

| Folder | Recognized formats/names |
|---|---|
| `data/raw/pubmed/` | `*.xml`, `*.xml.gz`, normalized `*.jsonl`, `*.jsonl.gz`. |
| `data/raw/openfda/` | Drugs@FDA `*.json` or `*.json.zip` bulk payloads. |
| `data/raw/b3db/` | Classification `*.tsv[.gz]` with the B3DB schema. |
| `data/raw/repodb/` | Original-schema `*.csv` / `*.tsv`. |
| `data/raw/local/` | Names ending `entities.jsonl`, `documents.jsonl[.gz]`, `judgments.jsonl`, `aliases.csv`. |

Subdirectories are scanned. For example, `team.documents.jsonl` is recognized, while arbitrary
`notes.jsonl` in the local folder is not. JSONL means one complete JSON object per UTF-8 line.
Unknown fields are rejected to expose spelling mistakes and schema drift.

### Local entity and document example

The following examples are **fictional format examples**, not scientific records. First add
`data/raw/local/team.entities.jsonl`:

```json
{"id":"drug:example-compound","kind":"drug","name":"Example compound","aliases":["Example brand"],"sources":["manual:format-example"]}
```

Then add `data/raw/local/team.documents.jsonl`:

```json
{"id":"local:example-paper","title":"Fictional format example","abstract":"Example compound and Alzheimer disease are mentioned in this fictional record.","entity_ids":["drug:example-compound","disease:alzheimer"],"source":"synthetic-format-example"}
```

Documents require `id` and a nonempty `abstract`. Optional fields include title, publication
date/precision, source, URL, DOI, MeSH/chemical metadata, and explicit canonical mentions. Explicit
IDs must exist. Missing dates are allowed for ordinary retrieval and excluded by cutoff filtering.

The fictional drug has no approval evidence and will therefore be excluded from approved-only drug
results. Supply approval/BBB metadata only when independently supported, recording its source;
do not invent metadata just to make a result appear.

After adding real inputs:

```bash
drugir run
# Include dense document encoding/revalidation if needed:
drugir run --dense
```

### Aliases and metadata merges

An `*aliases.csv` file adds alternative names to existing IDs:

```csv
entity_id,alias
drug:example-compound,A reviewed alternative name
```

Aliases are explicit mappings, not fuzzy matching rules. An alias added after external catalogues
load does **not** merge two already-existing canonical IDs or move their metadata. When a reviewed
mapping should unify source names before importing FDA/repoDB, seed a local entity with those
aliases first; local entities load before those catalogues. Inspect the resulting IDs/report.

This matters for base/salt splits such as the current abacavir records. Review chemical equivalence
before mapping, and keep judgment/explicit document IDs aligned with the chosen canonical record.

### Judgment example

Independent judgments go in a separate `*judgments.jsonl` file. This is another fictional example:

```json
{"drug_id":"drug:example-compound","disease_id":"disease:alzheimer","relevance":0,"source":"synthetic-format-example","status":"example"}
```

`relevance` is binary. No record means unjudged. IDs must exist with the correct drug/disease types.
For real curation, `source` should identify the evidence supporting the judgment. Do not import
the formatting examples into scientific evaluation.

Full schemas, source columns, and temporal fields are in [data-contracts.md](docs/data-contracts.md).

## Retrieval and scoring

| Method | Implementation | Role |
|---|---|---|
| `bm25` | `rank_bm25.BM25Okapi` over title/abstract/MeSH tokens. | Lexical baseline. |
| `dense` | Normalized PubMedBERT embeddings, exact cosine scoring. | Biomedical semantic matching. |
| `hybrid` | Reciprocal rank fusion of lexical and dense paper lists. | Combine lexical and semantic evidence. |
| `cross-encoder` | Hybrid pool rescored with the configured MS MARCO model. | Joint query/paper relevance baseline, currently general-domain. |
| `abc` | Shared non-entity MeSH descriptors connecting separate paper sets. | Indirect A–B–C literature association baseline. |
| `random` | Seeded entity ordering over the full eligible catalogue. | Chance baseline, including entities without papers. |

### Paper retrieval

BM25 uses the lexical tokenizer in `indexing.py`. Neural models use their own tokenizers and take
raw title/abstract strings. Do not substitute stopword-filtered BM25 tokens for neural input.
Long text is truncated at the configured sequence limit; v1 does not chunk documents.

Normalized document/query vectors make their dot product cosine similarity. BM25 and cosine have
different score scales, so hybrid fusion uses ranks: every document occurrence contributes
`1 / (rrf_k + rank)`, with one-based rank. Contributions from both lists are summed and the fused
pool is bounded by `candidate_documents`.

The cross-encoder scores only that bounded hybrid pool. A sigmoid is applied before entity
aggregation, preserving order without establishing calibrated probabilities. It scores query/paper
relevance; it does not verify the drug/disease treatment relationship.

### Paper scores become entity scores

Direct methods require both entities in a selected paper. Aggregate scores using:

- `max`: strongest paper; default, and a shared paper can cause tied entity scores.
- `sum`: all matched paper scores; can reward entities with larger literature volume.
- `top-mean`: mean of the strongest up to `aggregation_top_n` papers.

The aggregate is divided by the maximum entity aggregate for this query when positive; otherwise
normalized values are zero. A pre-feasibility value of 1 means highest relative evidence score in
this query. It is neither a probability nor a globally comparable confidence measure.

### Feasibility scoring

With feasibility enabled for a drug/CNS-disease pair:

```text
final score = normalized entity retrieval score + BBB adjustment

BBB positive      → +bbb_weight
BBB negative      → -bbb_weight
Unknown/conflict  → 0
```

The same drug/disease-role rule works in both entity directions. A free-text mechanism query
without a disease context gets no pair-specific adjustment. Default weight is 0.15; final scores
can exceed one or become negative.

BBB evidence does not establish efficacy. An adverse-event mention of a CNS disease can currently
be promoted too, which is why relationship-aware scoring remains a priority.

### Evidence output

`evidence_count` counts matched papers in the retrieved pool. Only `evidence_per_entity` strongest
references are returned. They include document ID, title, date, URL, score, and an extracted
sentence. The snippet maximizes overlap with the original query and is truncated at 700 characters;
it is not generated text or a candidate-specific relationship explanation.

ABC references are candidate-side bridge papers. Random results can have no papers. Read actual
paper context before interpreting any candidate as therapeutically supported.

## Configuration reference

Copy `configs/default.yaml` to `configs/<experiment>.yaml` for a new experiment. Every command
accepts `--config`. Omitted fields use `config.py` defaults; unknown/invalid values fail validation.

**Path rule:** relative paths resolve from the configuration's grandparent. For
`configs/experiment.yaml`, that is the repository root. Putting a file deeper under
`configs/team/experiment.yaml` changes the anchor. Keep configs directly under `configs/` or use
absolute data paths. From another working directory, pass an absolute config filename.

### Paths and catalogue

| Setting | Default | Meaning |
|---|---|---|
| `schema_version` | `1` | Root schema marker. |
| `paths.raw` | `data/raw` | Original source inputs. |
| `paths.processed` | `data/processed` | Generated canonical records/report. |
| `paths.index` | `artifacts/index` | Snapshot and optional dense matrix. |
| `paths.runs` | `runs` | Evaluation outputs and `LATEST` pointer. |
| `diseases` | Default YAML list | Scoped canonical diseases, aliases, identifiers, CNS status. |

Use separate processed/index/run locations for isolated experiments. Keep canonical entity IDs
stable across source updates whenever possible.

### Acquisition settings

| Under `acquisition` | Default | Meaning |
|---|---|---|
| `pubmed_per_disease` | `100` | Maximum selected PMIDs per job, including extra queries; 1–9,999. |
| `pubmed_batch_size` | `100` | XML records per efetch batch; 1–500. |
| `pubmed_start_year` / `pubmed_end_year` | `1990` / `2026` | Inclusive acquisition publication-year window. |
| `pubmed_sort` | `relevance` | Search ordering; alternatively `pub date`. |
| `pubmed_extra_query` | Therapy/repurposing expression | Additional restriction on configured disease searches. |
| `pubmed_queries` | Abacavir search in default YAML | Named extra PubMed expressions; keys become filename prefixes. |
| `b3db_url` | Pinned source URL | Exact BBB classification snapshot. |
| `repodb_article_id` | `3811674` | Original archived Figshare final database. |

Acquisition years control downloaded coverage. `evaluation.cutoff_year` separately filters the
search snapshot; it cannot recover old papers that were never acquired.

### Retrieval settings

| Under `retrieval` | Default | Meaning |
|---|---|---|
| `candidate_documents` | `200` | First-stage paper pool size. |
| `evidence_per_entity` | `3` | Maximum returned paper excerpts per entity. |
| `aggregation` | `max` | `max`, `sum`, `top-mean`. |
| `aggregation_top_n` | `3` | Number averaged for top-mean. |
| `bm25_k1` / `bm25_b` | `1.5` / `0.75` | Frequency saturation and length normalization. |
| `rrf_k` | `60` | Reciprocal rank fusion offset. |
| `dense_model` | `NeuML/pubmedbert-base-embeddings` | SentenceTransformer name or supported local model path. |
| `dense_revision` | `null` | Explicit model pin, if desired; resolved commit is recorded with vectors. |
| `cross_encoder_model` | `cross-encoder/ms-marco-MiniLM-L6-v2` | Joint query/paper scorer. |
| `cross_encoder_revision` | `null` | Optional reranker model revision. |
| `device` | `cpu` | `cpu`, `mps`, `cuda`. |
| `batch_size` | `16` | Neural encoding/prediction batch size. |
| `max_sequence_length` | `512` | Token truncation limit; must fit the actual selected model. |
| `approved_only` | `true` | Require known approval evidence for candidate drugs. |
| `bbb_weight` | `0.15` | Heuristic positive/negative BBB adjustment magnitude. |
| `seed` | `650` | Random ranking and bootstrap seed. |

Numeric configuration validation does not guarantee that every model supports every sequence
length/device. Changing approval filtering changes the benchmark population; record it explicitly.

### Evaluation settings

| Under `evaluation` | Default | Meaning |
|---|---|---|
| `ks` | `[5, 10, 20]` | Final entity-rank cutoffs; each 1–100. |
| `bootstrap_samples` | `1000` | Query bootstrap resamples; zero disables intervals. |
| `cutoff_year` | `null` | Optional literature/approval availability bound. |
| `mode` | `indication-recovery` | Known-indication recovery or `temporal-repurposing`. |

## Commands and everyday workflows

Use `drugir COMMAND --help` for options. Command docstrings describe prerequisites and side effects.

| Command | Reads | Writes / effects |
|---|---|---|
| `fetch` | Source config, optional credentials, download manifests | Raw exports/manifests; network access. |
| `prepare` | Raw folders and configured catalogue | Canonical records and ingestion report. |
| `index` | Prepared documents/entities, defining settings | Lexical snapshot; `--dense` adds/checks vectors. |
| `run` | Raw inputs; optional source endpoints | Prepare + index; `--download` first acquires sources. |
| `search` | Current snapshot/fingerprints, optional weights | JSON to terminal or `--output`. |
| `evaluate` | Index, judgments, selected methods | New experiment folder and `LATEST`. |
| `doctor` | Manifests/report | Read-only readiness inventory. |
| `serve` | Index/settings | Local server; model loading on neural use. |

### Acquire or refresh individual sources

```bash
drugir fetch --source pubmed
drugir fetch --source openfda
drugir fetch --source b3db
drugir fetch --source repodb

# Deliberately refresh an upstream selection, then rebuild its derived state.
drugir fetch --source pubmed --refresh
drugir run --dense
```

Changing diseases or sample size requires acquisition to obtain the added scope. `prepare` alone
cannot retrieve more literature.

### Investigate one stage at a time

```bash
drugir fetch
drugir prepare
drugir index
drugir index --dense
drugir doctor
```

Inspect `data/processed/ingestion-report.json` after preparation. Missing mentions or failed
approval/BBB joins there cannot be repaired simply by selecting a different ranker.

### Search variants

```bash
drugir search "Alzheimer's" --method bm25 --k 5
drugir search "Alzheimer's" --method hybrid --feasibility --k 5
drugir search abacavir --direction drug-to-disease --method hybrid
drugir search "reverse transcriptase inhibition" --direction mechanism --method hybrid
drugir search "Alzheimer's" --method cross-encoder --output runs/alzheimer-search.json
```

Quote names containing spaces/apostrophes. A known canonical ID also works:
`drugir search disease:alzheimer --method bm25`. Entity directions reject unknown/ambiguous
names; use mechanism direction for free text rather than expecting fuzzy entity resolution.

### Evaluate a subset or the full query set

```bash
# Stable-prefix smoke test, not a representative random sample.
drugir evaluate --methods random,bm25,abc --max-queries 4

# Full neural comparison; build the dense index first.
drugir evaluate --methods random,bm25,bm25+feasibility,abc,dense,hybrid,hybrid+feasibility

# Separate directions when comparing different candidate universes.
drugir evaluate --methods bm25,hybrid --directions disease-to-drug
drugir evaluate --methods bm25,hybrid --directions drug-to-disease

# Preserve a specifically named experiment in a NEW directory.
drugir evaluate --methods bm25,hybrid --output runs/my-new-experiment
```

Default evaluation methods are `random,bm25,bm25+feasibility,abc`. Search uses the flag
`--feasibility`; evaluation appends `+feasibility` to a method name. `--max-queries` uses the
first stable-sorted queries. Existing completed summaries are not overwritten.

### Separate experiment configuration

```bash
cp configs/default.yaml configs/experiment.yaml
# Edit settings and give the experiment distinct output paths if appropriate.
drugir run --config configs/experiment.yaml --dense
drugir search "Alzheimer's" --config configs/experiment.yaml --method hybrid
drugir evaluate --config configs/experiment.yaml --methods bm25,hybrid
```

## Reading search results

The JSON response is a `SearchResponse`. This is a **shortened fictional example**:

```json
{
  "query": "Brain disease",
  "direction": "disease-to-drug",
  "method": "hybrid",
  "resolved_entity": "disease:brain",
  "results": [
    {
      "entity": {"id": "drug:alpha", "kind": "drug", "name": "Alpha"},
      "score": 1.15,
      "retrieval_score": 1.0,
      "feasibility_adjustment": 0.15,
      "evidence_count": 4,
      "evidence": []
    }
  ],
  "warnings": []
}
```

Paper references and default entity fields are omitted here for readability. Normal evidence-backed
results include actual paper details in `evidence`.

| Field | Meaning |
|---|---|
| `resolved_entity` | Canonical query match; may be null for mechanism search. |
| `entity` | Candidate identity, aliases, external IDs, approval/BBB metadata. |
| `retrieval_score` | Within-query aggregate before feasibility. |
| `feasibility_adjustment` | Inspectable BBB contribution. |
| `score` | Final ordering value, not a probability. |
| `evidence_count` | Supporting papers in the retrieved candidate pool. |
| `evidence` | Bounded strongest paper references/excerpts. |
| `warnings` | Missing coverage and relevant qualifications. |

An empty list may mean no eligible co-mentioned candidate exists in the selected pool. It does
not establish that a scientific relationship is impossible.

## Evaluation and interpretation

### Labels and query population

Default indication recovery maps exact scoped repoDB aliases as follows:

- Approved → positive (1).
- Terminated, Suspended, Withdrawn → benchmark negative (0).
- Pair without a judgment → unjudged.
- Pair with both binary outcomes → excluded and counted as conflicting.

A failed-trial status does not prove pharmacological impossibility. An approved indication alone
does not establish a historically verified repurposing event. Subtypes are not automatically
promoted into broader configured parent diseases.

Queries without positives are excluded from Recall/MRR/nDCG. For included queries, all scoped
positives remain in denominators, even when absent from the corpus or filtered by approval.
Eligibility and coverage are separate diagnostics, not reasons to silently remove difficult labels.

### Metrics and a worked example

| Metric | Definition |
|---|---|
| `recall@k` | Retrieved positives / every judged positive for the query. |
| `mrr@k` | Reciprocal rank of the first positive within k, otherwise zero. |
| `ndcg@k` | Binary gains discounted by log2(rank+1), normalized by ideal DCG. |
| `unjudged@k` | Share of returned entities without judgments; unknowns score zero in ranking metrics. |
| `returned@k` | Actual result count, exposing short/empty lists. |
| `non_bbb@k` | Known BBB-negative share for CNS disease-to-drug queries. |
| `unknown_bbb@k` | Missing/conflicting BBB share for those CNS queries. |

Fractions over returned results use actual returned count. Empty lists record zero alongside
`returned@k = 0`; do not read that as proof of perfect feasibility.

Suppose a query has four positives and the top five retrieves two, at ranks 2 and 4. Recall@5 is
`2/4 = 0.5`; MRR@5 is `1/2 = 0.5`. The nDCG numerator has gains at 2 and 4, while ideal DCG places
all four positives at 1–4. A positive absent from every indexed paper still counts among the four.

Means weight queries equally. Joint means do not give equal total weight to directions when one
has more queries. Intervals resample query values with replacement using a fixed seed. Paired
comparisons subtract BM25 on each same query before bootstrapping differences. Disease clustering
and shared pairs across directions are not modeled as independent clusters.

### Run outputs

| File | How to use it |
|---|---|
| `summary.json` | Direction/joint metrics, intervals, paired deltas, exclusions, hashes, package/model metadata. |
| `per-query.csv` | Inspect individual successes, failures, and coverage. |
| `queries.jsonl` | Reconstruct the fixed query set, labels, and positive coverage counts. |
| `rankings.jsonl` | Inspect entities and papers behind each method's ranking. |
| `config.json` | Recover resolved experiment settings and paths. |
| `README.md` | Convenience result table; detailed evidence remains in JSON/CSV. |

`runs/LATEST` is a text pointer to the newest run path, not another directory. Outputs are local
and git-ignored by default. Share frozen settings and selected results explicitly when collaborating,
respecting the underlying source terms.

Query coverage fields are `relevant_total`, `relevant_eligible`, `relevant_in_corpus`, and
`relevant_direct_evidence`. They distinguish labels, eligibility, mention coverage anywhere, and
direct query/candidate co-occurrence.

### Initial recorded snapshot

The November 8, 2026 development snapshot contains 1,922 abstracts, 20 configured diseases, 2,504
catalogue drugs with approval evidence, and 178 nonconflicting judged pairs (127 positive). There
are 12 disease queries and 125 drug queries with usable positives.

| Method | Disease → drug Recall@10 | Drug → disease Recall@10 |
|---|---:|---:|
| BM25 | 0.2792 | 0.5640 |
| Dense | 0.2978 | 0.5320 |
| Hybrid | 0.3044 | 0.5720 |
| Hybrid + BBB | 0.2863 | 0.5720 |

These are historical baselines, not expected outcomes after every refresh. Reverse random Recall@10
was already 0.524 because ten of only twenty diseases are returned. Joint means are dominated by
125 reverse queries. Dense is not consistently better than BM25; BBB adjustment did not improve
Recall@10 in this snapshot.

Of 127 positive pairs, 66 had eligible approval metadata on their drug record and 72 had direct
literature co-occurrence. Normalization and coverage are important bottlenecks. The cross-encoder
has a four-query evaluation smoke test, not a full 137-query benchmark. Details are in
[v1-validation.md](docs/v1-validation.md) and [methodology.md](docs/methodology.md).

### Temporal mode

`cutoff_year` bounds indexed papers and requires drug approval evidence no later than that year.
That is not by itself future-repurposing evaluation.

`mode: temporal-repurposing` additionally needs reviewed local judgments with:

- `confirmed_date`: actual outcome after the cutoff.
- `prior_indication_date`: actual earlier indication event no later than cutoff.
- `prior_indication`: verified different indication, not the target disease repeated.

Missing dates are excluded, not replaced by download/registration dates. Prior-name checks depend
on registry resolution; manual historical review remains required. Archived repoDB alone lacks
these facts. Current BBB labels and pretrained models are not historically frozen, so a date-filtered
corpus is not a complete historical reconstruction.

## Python and HTTP interfaces

### Direct Python use

```python
from drug_repurpose_ir.config import load_config
from drug_repurpose_ir.engine import SearchEngine

settings = load_config("configs/default.yaml")
engine = SearchEngine(settings)  # Requires a prepared, compatible index.
response = engine.search(
    query="Alzheimer's",
    direction="disease-to-drug",
    method="bm25",
    k=5,
    feasibility=False,
)
for result in response.results:
    print(result.entity.name, result.score, result.evidence_count)
    for paper in result.evidence:
        print("  ", paper.document_id, paper.title, paper.url)
```

Reuse an engine for multiple queries to reuse weights/caches. Treat its settings/snapshot as fixed;
create a new engine after rebuilding. To run preparation/indexing programmatically:

```python
from drug_repurpose_ir.config import load_config
from drug_repurpose_ir.ingestion import ingest
from drug_repurpose_ir.indexing import build_index

settings = load_config("configs/default.yaml")
report = ingest(settings)                  # Existing raw files only.
manifest = build_index(settings, dense=False)
print(report["documents"], manifest["index_hash"])
```

### HTTP backend

```bash
drugir serve --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs` for interactive request schemas/examples.

| Route | Purpose |
|---|---|
| `GET /health` | Loaded document count/index hash; no retrieval inference. |
| `GET /entities?kind=disease&q=alzheimer&limit=20` | Preferred-name substring lookup, not every alias. |
| `POST /search` | The same validated search engine used by CLI/Python. |

From a second terminal:

```bash
curl http://127.0.0.1:8000/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"Alzheimer disease","direction":"disease-to-drug","method":"bm25","k":5,"feasibility":false}'
```

Expected input/schema errors return 422; expected runtime availability errors return 503. One
engine is reused and search requests are serialized by a lock protecting lazy model/caches.
Restart after rebuilding. This is a local research backend; there is no Next.js frontend, and
teammates do not need frontend code to run experiments.

## Reproducibility, caches, and hardware

### Fingerprints

Lexical compatibility includes document/entity file hashes, cutoff year, and BM25 settings. Dense
compatibility adds the index hash, model name, configured revision, and sequence limit. Metadata
also records resolved remote model commits when available; dense query encoding reuses that
snapshot unless configuration explicitly pins another revision.

Loaders reject incompatible snapshots instead of silently mixing new data with old indexes.
Entity metadata is part of the index hash, so even an entity-only change currently causes a
conservative full dense rebuild.

Arbitrary **Python code changes are not automatically fingerprinted**. After modifying tokenization,
preprocessing, or embedding behavior, use a fresh experiment index path and rebuild; an unchanged
old manifest is not proof it detects a changed algorithm. Record code revision when sharing runs.

### Cache layers

| Artifact | Purpose | Update mechanism |
|---|---|---|
| Raw exports/manifests | Reuse acquired bytes with validated request identity/checksum. | `fetch --refresh` explicitly bypasses reuse. |
| Prepared JSONL | Canonical stage contracts. | `prepare` or `run`. |
| Lexical snapshot | Selected documents/catalogue/tokens. | `index` rebuilds on defining fingerprint changes. |
| Dense matrix | Compatible document vectors. | `index --dense` / `run --dense`. |
| Model cache | Retain pretrained weights. | Model settings/revisions select the cached/downloaded snapshot. |
| In-memory caches | Reuse work during one engine/server lifetime. | Empty in a fresh engine/process. |

Lexical search performs no source download. Neural use may access the model hub for initial
weights/resolution. Once all necessary files are cached, `HF_HUB_OFFLINE=1` intentionally requires
local weights.

### Mac and GPU server

CPU is the default. With compatible Apple Silicon/PyTorch, try `device: mps` in a separate config.
For a CUDA server use `device: cuda` and a suitable server-side PyTorch install. Recreate its
environment rather than copying Mac binaries.

For Great Lakes, adapt scheduler/account/partition/module settings to the team's allocation; this
repository does not submit jobs. Configure storage paths and an approved `HF_HOME` cache as needed.
Reduce `batch_size` on memory exhaustion. V1 holds catalogues/documents/BM25 state in memory and
loads full JSON export payloads; exact vector search is a starter-scale choice.

`requirements-lock.txt` records exact initial Mac versions. `pyproject.toml` is the portable
dependency contract; GPU/platform-specific installation may resolve different transitive packages.

## Testing and contributing

### Standard checks

```bash
python -m pip install -e '.[neural,api,dev]'
pytest -q
ruff check src tests
ruff format --check src tests
python -m compileall -q src
python -m pip check
```

Use `ruff format src tests` when intentionally formatting edits, then rerun checks. The initial
suite has 13 tests; fixtures create fictional inputs/indexes in temporary directories. They do not
touch real datasets or download model weights.

### Tests as worked examples

- `conftest.py` builds fictional positive/negative/unknown BBB cases, unapproved drugs, an absent
  judged positive, and papers on both sides of date cutoffs.
- `test_ingestion.py` checks source joins, brand linking, deduplication, label isolation, download
  cache integrity/refetch, secret redaction, and missing-corpus failure.
- `test_pipeline.py` checks hand-computable metrics, alias ambiguity/boundaries, both directions,
  BBB rules, deterministic random ranking, fake-encoder vector alignment, date/stale checks,
  saved coverage, mixed PubMed XML, and optional HTTP validation.

One historical downloader-test name mentions retry, but that fixture checks successful requests,
reuse, and corruption-triggered refetch rather than backoff branches. Real neural smoke checks
are documented separately from the offline test suite.

### Extension map

| Change | Main files | What to preserve/check |
|---|---|---|
| New source | `acquisition.py`, `ingestion.py` | Raw provenance, explicit adaptation, canonical schemas, tiny fixture. |
| Entity normalization | `entities.py`, reviewed local mappings | Ambiguity, chemical forms, stable IDs, linked labels. |
| PubMed parsing | `pubmed.py` | Nested text, unknown/imprecise dates, source IDs, cutoff behavior. |
| Retrieval method | `engine.py`, possibly `indexing.py` | Add method to `SearchRequest.method` and evaluation's allowed set; preserve row alignment. |
| Entity aggregation | `aggregate()` and retrieval config | Literature-volume effects and comparisons on identical labels. |
| Feasibility feature | Metadata schema and pair scoring/output | Explicit unknowns, source evidence, inspectable adjustments. |
| Metric | `ranking_metrics()` and summaries | Hand-computable values, denominators, short/empty lists. |
| Command or API | `cli.py`, `api.py` | Reuse the engine contract, update help, verify request handling. |

Keep stage responsibilities clear and pass `Settings` explicitly. Document arguments, returns,
side effects, failure conditions, and research assumptions. Explain processing steps in inline
comments so a new teammate can follow a function without reconstructing the whole repository.

For experiments, freeze configuration, use a unique run folder, inspect query-level evidence, and
compare the same query set/candidate universe. Repeated tuning on the same judgments needs a
separate held-out test set before presenting generalization claims or training a learned reranker.

## Troubleshooting

| Symptom | Explanation / next action |
|---|---|
| `drugir` not found | Activate `.venv`, install editable package, or use `.venv/bin/drugir`. |
| Config not found | Run from repository root or pass an absolute `--config`. |
| No documents | Acquire/import recognized literature, then `prepare` or `run`. |
| Missing index | Run `run`, or `prepare` then `index`. |
| Stale index | Rebuild after data/defining-settings changes; restart existing API process. |
| Missing/stale dense index | Install neural extra and run `index --dense` for the same config. |
| Offline model error | Initial weights must be downloaded before requiring local-only operation. |
| MPS/CUDA failure or memory exhaustion | Establish CPU baseline, verify device environment, lower batch size. |
| Unknown/ambiguous query | Use canonical ID, review aliases, or mechanism mode for free text. |
| Resolved query but empty results | Inspect mentions, approval eligibility, paper depth, direct co-occurrence. |
| Familiar drug filtered | Exact record may lack approval evidence because of a base/salt split; review joins. |
| Unknown/conflicting BBB | Review B3DB name joins/observations; no predicted replacement is fabricated. |
| No temporal judgments | Curate actual outcome/prior-indication dates or use indication recovery. |
| Output already exists | Choose a new `--output` directory to preserve completed runs. |
| Relevant papers, poor candidates | Inspect adverse/background/comparator mentions and shared max-paper scores. |

Debug in this order: `doctor` → ingestion report → a single search's evidence → per-query results
and manifest. Inspect coverage before attributing every error to the retrieval model.

## Known limitations and next steps

The main next priorities are:

1. **Catalogue reconciliation:** review salt/brand/chemical identifiers, broad chemical names,
   and ambiguities. The current abacavir/base-versus-sulfate split is a concrete gap.
2. **Relationship-aware evidence:** separate treatment from symptoms, adverse events, background
   chemistry, alternative regimens, and incidental mentions.
3. **Coverage:** expand targeted literature and examine per-disease/per-pair gaps in the bounded sample.
4. **Study design:** curate dated repurposing events, prior different indications, and separate
   development/test queries before extensive tuning or learned reranking.
5. **Experiments:** compare biomedical encoders/rerankers, aggregation, depth, indirect bridges,
   and feasibility ablations against frozen baselines.
6. **Scale/usability:** add chunking/approximate search when justified; integrate a frontend through
   the existing API if useful.

The abacavir–Alzheimer's anchor is not recovered by current sampling/linking. Some Alzheimer's
candidates come from symptom-management papers, while abacavir can return stroke because a paper
measured cardiovascular events during HIV treatment. These are explicit failure cases motivating
the next iterations, not successful repurposing discoveries.

Source/model references and attribution are in [sources.md](docs/sources.md). Abstract reuse depends
on publisher/source terms; datasets retain their own licenses. Read [methodology.md](docs/methodology.md)
for the study limitations and [roadmap.md](docs/roadmap.md) for the continuation plan.
