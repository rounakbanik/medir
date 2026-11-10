# Code and configuration

Paths and commands below are relative to the project root.

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
| [ir/](../src/drug_repurpose_ir/ir) | Packaged reference algorithms; original code and documentation. |
| [reference_adapter.py](../src/drug_repurpose_ir/reference_adapter.py) | Input JSONL, integer ID mapping, model setup, and metric-list padding. |
| [cli.py](../src/drug_repurpose_ir/cli.py) | Commands and pipeline orchestration. |
| [models.py](../src/drug_repurpose_ir/models.py) | Validated records used on disk and in responses. |
| [config.py](../src/drug_repurpose_ir/config.py) | YAML settings, defaults, and path resolution. |
| [acquisition.py](../src/drug_repurpose_ir/acquisition.py) | Downloads, retries, checksums, and reuse of local files. |
| [ingestion.py](../src/drug_repurpose_ir/ingestion.py) | Source adapters, catalogue joins, deduplication, and coverage report. |
| [pubmed.py](../src/drug_repurpose_ir/pubmed.py) | PubMed XML and publication dates. |
| [entities.py](../src/drug_repurpose_ir/entities.py) | Canonical IDs, aliases, and mention matching. |
| [indexing.py](../src/drug_repurpose_ir/indexing.py) | Build the reference index, save embeddings, check compatibility. |
| [engine.py](../src/drug_repurpose_ir/engine.py) | Paper retrieval, entity scoring, BBB rule, and evidence selection. |
| [evaluation.py](../src/drug_repurpose_ir/evaluation.py) | Query sets, metrics, bootstrap intervals, and run outputs. |
| [io.py](../src/drug_repurpose_ir/io.py) | JSON/JSONL files and checksums. |
| [api.py](../src/drug_repurpose_ir/api.py) | HTTP access to the same search engine. |

`tests/` contains small, fictional examples. `examples/demo/` supplies the runnable demo.
`reference/` contains the original files. Runtime imports use the packaged copies in `ir/`.
`ir/provenance.json` records source hashes and the few import substitutions. Tests compare the
copies byte-for-byte after those substitutions; formatting tools exclude this directory.

## Configuration

Edit [configs/default.yaml](../configs/default.yaml), or copy it for an experiment:

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
[config.py](../src/drug_repurpose_ir/config.py). Changing the collection scope requires fetching
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

[data-contracts.md](data-contracts.md) covers file formats,
[sources.md](sources.md) lists datasets and attribution, and
[roadmap.md](roadmap.md) tracks the planned research work.
