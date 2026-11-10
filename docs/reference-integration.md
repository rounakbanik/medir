# Reference-code integration

The retrieval core now uses the files from `reference/starter-code`. The originals are
unchanged. Copies live in `src/drug_repurpose_ir/ir/` so installation does not depend on
the git-ignored reference folder.

## What is reused

| Original file | Use in this project |
|---|---|
| `document_preprocessor.py` | `RegexTokenizer` for the lexical corpus and queries. |
| `indexing.py` | `Indexer`, basic/positional indexes, postings, statistics, and save/load. |
| `ranker.py` | `Ranker`, all six lexical scorers, and `CrossEncoderScorer.score()`. |
| `vector_ranker.py` | `VectorRanker.query()` for query encoding and dot-product ranking. |
| `relevance.py` | `map_score()` and `ndcg_score()` for entity-ranking evaluation. |
| `l2r.py` | Packaged for later experiments; not trained or enabled in the pipeline. |
| `network_features.py` | Packaged for later experiments; no citation graph is built yet. |

Only imports between these files changed to package-relative imports. All other bytes,
including comments, docstrings, formulas, formatting, and line endings, are preserved.
`ir/provenance.json` records the source/copy hashes and each import substitution. Ruff
excludes these files, and a regression test checks the copies against that record and
the originals when available.

## What the adapters do

`reference_adapter.py` exports title, abstract, and MeSH text as the same `docid`/`text`
JSONL expected by the original Indexer. Integer IDs are rows in `documents.jsonl`;
the original PubMed/local IDs and metadata remain in those rows. The saved lexical index
uses the original JSON format under `artifacts/index/lexical/`.

Query adaptation removes unknown terms for TF-IDF and Dirichlet only: those scorers
divide by document frequency or collection probability. The scorer implementations
are unchanged. Negative BM25 scores are retained, including in the paper evidence.
Model setup supplies the configured device, sequence limit, and cached model snapshot.

The biomedical layer still prepares datasets, links entities, combines lexical/dense
ranks, aggregates paper scores into drug/disease scores, applies the BBB rule, and
builds the evaluation query set. Those operations are specific to this project rather
than Wikipedia document retrieval. The old `rank-bm25` backend has been removed.

MAP keeps the original denominator: `min(positives in the supplied ranking, k)`.
The supplied ranking here contains up to the largest configured evaluation cutoff
(20 by default). This differs from dividing by every judged positive, so MAP can be 1
when recall is below 1. nDCG gets zero-padded actual/ideal lists when either is shorter
than k. Both functions receive relevance labels, not model scores.

## Checks

Verified locally on November 9, 2026, with Python 3.12 on Mac CPU:

- 22 project tests passed, covering source preservation, index round trips, ranker
  parity, negative BM25 scores, unknown query terms, metrics, and pipeline/API behavior.
- The supplied index/tokenizer/scorer/relevance suite ran 66 tests: 55 passed and 11
  were skipped by its existing decorators. Model-downloading cases were omitted.
- Ruff checks and formatting passed for the surrounding code; dependency checks passed.
- The real index was rebuilt with 1,922 abstracts and 15,020 lexical terms. Dense
  embeddings were rebuilt, then all eleven variants below ran on the same 137 queries.
- A separate cross-encoder smoke evaluation used two queries and 20 candidate papers.
  This checks the integration, not comparative cross-encoder quality.
- Alzheimer's and abacavir returned results with BM25 and hybrid. The API's health and
  TF-IDF search routes also passed against the real corpus.

Run the project checks with `pytest -q`. To rerun the supplied tests, use
`HF_HUB_OFFLINE=1 python scripts/check_reference.py`; this needs the original tests.

## Evaluation results

There are 12 disease queries and 125 drug queries. Values are macro averages within
each direction. These measure known indication recovery with the current incomplete
corpus and judgments; they do not establish that a candidate is an effective treatment.

| Method | Disease → drug Recall@10 | MAP@10 | nDCG@10 | Drug → disease Recall@10 | MAP@10 | nDCG@10 |
|---|---:|---:|---:|---:|---:|---:|
| Random | 0.0000 | 0.0000 | 0.0000 | 0.5240 | 0.1602 | 0.2441 |
| BM25 | 0.2702 | 0.3471 | 0.3134 | 0.5640 | 0.5019 | 0.5151 |
| TF-IDF | 0.2824 | 0.2723 | 0.2837 | 0.5640 | 0.4666 | 0.4898 |
| Dirichlet | 0.2911 | 0.2750 | 0.2872 | 0.5640 | 0.4928 | 0.5083 |
| Pivoted | 0.2732 | 0.3162 | 0.2959 | 0.5640 | 0.4955 | 0.5102 |
| Word count | 0.2824 | 0.2550 | 0.2638 | 0.5640 | 0.4626 | 0.4868 |
| Custom | 0.2628 | 0.2858 | 0.2812 | 0.5640 | 0.4928 | 0.5083 |
| ABC | 0.0000 | 0.0000 | 0.0000 | 0.5760 | 0.4416 | 0.4744 |
| Dense | 0.2978 | 0.3116 | 0.3024 | 0.5320 | 0.4813 | 0.4920 |
| Hybrid | 0.2984 | 0.3156 | 0.3173 | 0.5720 | 0.5067 | 0.5210 |
| Hybrid + BBB | 0.2804 | 0.2980 | 0.3035 | 0.5720 | 0.5201 | 0.5310 |

Hybrid improves recall slightly over BM25. BBB reranking lowers disease-to-drug recall
and nDCG, while improving reverse-direction nDCG. This does not support a general claim
that BBB reranking improves retrieval. Reverse recall has a high random baseline because
there are only 20 disease candidates. Missing literature, approval joins, and salt/name
splits still limit recovery.

The complete local outputs, including rankings, configuration, per-query metrics, and
confidence intervals, are in `runs/reference-ir-v1/`. The smaller cross-encoder run is
in `runs/reference-cross-encoder-smoke/`. These generated folders are git-ignored.

To repeat the full comparison after building a dense index:

```bash
drugir evaluate \
  --methods random,bm25,tf-idf,dirichlet,pivoted,word-count,custom,abc,dense,hybrid,hybrid+feasibility \
  --output runs/reference-ir-v2
```

The November 8 results in `v1-validation.md` used the earlier library-based backend.
They are retained as history and should not be mixed with this run.
