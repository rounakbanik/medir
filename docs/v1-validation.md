# V1 verification — November 8, 2026

This is the earlier library-based baseline. The current reference-code integration and its
November 9 results are in [reference-integration.md](reference-integration.md).

The Python implementation was verified locally on Apple Silicon with Python 3.12. The assignment reference folder was left unchanged.

## Real data loaded

| Item | Count |
|---|---:|
| Configured diseases | 20 |
| Deduplicated PubMed abstracts | 1,922 |
| FDA application records read | 29,446 |
| Canonical drug entities | 3,315 |
| Drug entities with FDA approval evidence | 2,504 |
| Drug entities with matched BBB metadata | 1,041 |
| Normalized judgment records | 183 |
| Nonconflicting judged pairs | 178 |
| Positive pairs | 127 |
| Disease-to-drug evaluation queries | 12 |
| Drug-to-disease evaluation queries | 125 |

B3DB supplied 7,807 raw records and the archived repoDB supplied 10,800 rows. Only exact configured disease aliases enter the scoped judgments. Conflicting outcomes for one pair were excluded. The FDA adapter skipped combination products; missing approval matches remain unknown.

## Full benchmark

Seven variants ran on the same 137 queries. Values below are macro averages within each direction.

| Method | Disease → drug Recall@10 | Drug → disease Recall@10 | Joint nDCG@10 |
|---|---:|---:|---:|
| random | 0.0000 | 0.5240 | 0.2227 |
| bm25 | 0.2792 | 0.5640 | 0.4992 |
| bm25+feasibility | 0.2750 | 0.5640 | 0.4973 |
| abc | 0.0000 | 0.5760 | 0.4328 |
| dense | 0.2978 | 0.5320 | 0.4754 |
| hybrid | 0.3044 | 0.5720 | 0.5055 |
| hybrid+feasibility | 0.2863 | 0.5720 | 0.5116 |

The random reverse-direction baseline has only 20 disease candidates; k=10 therefore retrieves a large fraction by chance. The joint mean is dominated by the 125 reverse queries. BBB reranking did not improve Recall@10 in this run. These are development baselines, not evidence of novel drug repurposing.

Raw run: `runs/v1-verified/` (local, git-ignored). `summary.json` includes confidence intervals, paired deltas, exclusions, package versions, and model revision. `per-query.csv` and `rankings.jsonl` preserve individual outcomes and supporting papers.

## Coverage limits

Of 127 scoped positive pairs, 66 drugs had approval evidence under the current exact-name catalogue joins, and 72 pairs had direct literature co-occurrence in this sample. All positives stayed in metric denominators. Salt/ingredient reconciliation and corpus expansion are substantial opportunities.

## Other verification

- 13 automated tests pass, including golden metric values, source ingestion, alias ambiguity, BBB/approval behavior, bidirectional evidence, stale/dense alignment checks, temporal failures, deterministic outputs, download-cache checksums, and API validation.
- Ruff lint, format checks, Python compilation, and installed-package dependency checks pass.
- Re-running local ingestion preserved corpus/entity hashes and reused the index.
- The isolated synthetic offline demo ran and evaluated independently of real data.
- The real-corpus FastAPI health and search endpoints returned successful responses.
- Dense and cross-encoder models ran on actual text; cross-encoder search was verified in both directions.
- Cross-encoder evaluation and BBB reranking ran on a four-query smoke subset at full candidate depth (`runs/v1-cross-encoder-smoke/`); a full cross-encoder benchmark remains to be run.
- True temporal evaluation correctly refuses undated repoDB-only labels. It needs reviewed dated outcomes and prior different indications.

## Model snapshots

- Biomedical encoder: `b79526d6ef3645e0df4530322e266f24c829f5ef`.
- Cross-encoder: `233902d25c440f23af6f7d6e94d2946bac0bee0a`.

The dense query encoder reloads the exact snapshot used to encode documents. Models are cached under `artifacts/models/`.
