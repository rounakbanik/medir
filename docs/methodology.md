# Evaluation interpretation

MAP and nDCG now use the implementations copied from `reference/starter-code/relevance.py`.
The formulas, docstrings, and comments in that module are unchanged. The adapter pads short
nDCG input lists with zeros. MAP keeps its original denominator: the number of positive gains
in the supplied ranking, capped at k. It does not divide by all positive judgments, so a high
MAP can coexist with low recall. Per-query `map@k` values are averaged for the run summary.

## What v1 measures

The default benchmark measures retrieval of **known drug indications** from a sampled literature
corpus. An approved repoDB indication is not by itself a verified repurposing event: proving that
requires evidence that the same drug was previously approved for a different indication.

Drug/disease dictionaries may use names and identifiers from repoDB, but no repoDB relationship,
status, or outcome is injected into a document or retrieval score. Judgments remain separate from
the index. FDA approval eligibility is sourced independently from Drugs@FDA or explicit curation.
Vocabulary coverage from repoDB is disclosed; an eventual prospective benchmark should freeze the
vocabulary too.

The benchmark candidate universe includes all eligible canonical entities in scope, even when
they have no papers. Recall denominators include every scoped positive judgment. Coverage columns
show eligible positives, positives mentioned anywhere in the corpus, and positives with direct
query/candidate co-occurrence. This exposes retrieval ceilings instead of silently scoring only
easy covered pairs.

## Ranking metrics

For each query, deduplicate entity IDs and evaluate the first k results. Recall@k divides retrieved
positives by **all** judged positives for that query. MRR@k is the reciprocal rank of the first
positive within k, or zero. Binary nDCG@k uses log2(rank+1) discounts and an ideal ranking with
min(k, number of positive judgments) positives. Negative and unjudged candidates both score zero.
Unjudged@k is the fraction of **returned** results without a judgment; returned@k makes short lists
visible. Queries without positives do not enter these metrics.

Means are macro averages over queries, separately by direction and jointly. Joint means weight
queries equally; they do not give equal total weight to directions. The two directions share
underlying pairs, so a joint interval does not model their dependence. Bootstrap intervals resample
queries with replacement. Differences against BM25 use paired per-query values. Samples, seeds,
query sets, data hashes, and configuration are retained. Intervals from a tiny disease/label set
should not be interpreted as broad generalization. No tuning/train split is claimed in v1, and
repeated tuning on these labels will require a separate held-out test set.

For CNS disease-to-drug queries, non_bbb@k counts known negative BBB labels divided by returned
results. unknown_bbb@k separately counts missing/conflicting labels. These metrics are descriptive,
not proof of exposure at a therapeutic dose. Reverse-query outputs use the same BBB pair rule,
but the CNS summary metric is restricted to disease-to-drug queries.

## Time splits

`cutoff_year` bounds indexed publication dates at December 31. Missing dates are excluded.
Month/year-only dates are conservatively represented by their final day. Candidate drugs require
a known approval date no later than the cutoff. Approval dates are earliest observed original
application dates in the supplied Drugs@FDA snapshot and can be incomplete.

`temporal-repurposing` mode also requires reviewed `confirmed_date`, `prior_indication_date`, and
`prior_indication` fields on evaluation pairs. Missing dated labels produce a clear error rather
than an invented split. Conflicting outcome labels are excluded and counted. A trial termination
is a source status used as a benchmark negative, not conclusive evidence of pharmacological failure.

This is not a completely historical reconstruction. The default neural encoder and cross-encoder
may have seen later publications; B3DB labels and dictionary aliases are current snapshots without
per-observation availability dates. BBB reranking remains retrospective even if papers are date
filtered. Claiming prospective discovery requires historical source/model snapshots or a clearly
qualified retrospective design.

## Known limitations and next experiments

Dictionary linking is conservative but still susceptible to broad drug names, ambiguous short
forms, salt forms, and background chemical mentions. The v1 direct rankers require co-occurrence,
so they can miss indirect mechanistic evidence. ABC supplies a basic indirect comparator using
MeSH bridges, not a validated causal explanation. The generic cross-encoder is a baseline only.
Missing dates, sparse repoDB disease coverage, and PubMed query sampling can limit achievable recall.

The system retains evidence for inspection. It does not run docking, make efficacy claims, infer
BBB labels, generate scientific citations, or synthesize clinical advice.

## Running evaluation

Build the index first, then run the methods you want to compare:

```bash
drugir evaluate --methods bm25,hybrid,hybrid+feasibility
drugir evaluate --methods random,bm25,abc --max-queries 4
drugir evaluate --methods bm25,hybrid --directions disease-to-drug --output runs/experiment-01
```

Hybrid needs dense embeddings. Search uses `--feasibility`; evaluation uses the
`+feasibility` suffix. `--max-queries` selects a stable prefix for a quick check.
Choose a new named output directory for each run; completed summaries are not overwritten.

Each run writes the following under `runs/`:

| File | Contents |
|---|---|
| `summary.json` | Metrics, intervals, exclusions, and model/data metadata. |
| `per-query.csv` | Individual query scores and coverage counts. |
| `queries.jsonl` | The query set and judgments used. |
| `rankings.jsonl` | Ranked entities and supporting papers for every variant. |
| `config.json` | Resolved settings for the run. |

`runs/LATEST` is a text file pointing to the latest evaluation directory.

Current results are in [reference-integration.md](reference-integration.md).
