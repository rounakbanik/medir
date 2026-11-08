"""Evaluate entity rankings against independent drug/disease pair judgments.

This module owns metric definitions, label exclusions, reproducible query
sets, query-level bootstrap intervals, paired comparisons, and run artifacts.
Missing relevant entities stay in recall denominators. The default task is
known indication recovery, not a claim of prospective repurposing discovery.
"""

import csv
from collections import defaultdict
from datetime import UTC, date, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

from .engine import SearchEngine
from .io import atomic_text, digest, read_jsonl, write_json, write_jsonl
from .models import Judgment


def ranking_metrics(ranked: list[str], judgments: dict[str, int], ks: list[int]):
    """Calculate binary Recall, reciprocal rank, nDCG, and coverage at each k.

    Args:
        ranked: Entity IDs in retrieved order; duplicates are removed stably.
        judgments: Entity ID to binary relevance for this one query.
        ks: Positive rank cutoffs, validated by the configuration loader.

    Returns:
        A flat mapping with recall@k, mrr@k, ndcg@k, unjudged@k, and returned@k.

    Recall divides by all positive judgments, including positives never returned.
    nDCG uses binary gains and log2(rank+1) discounting. Unjudged entities score
    zero relevance but are counted separately among returned results. Empty
    result lists produce zero metrics, while queries with no positives are invalid.

    Raises:
        ValueError: The query has no positive judgments.
    """
    # Stable deduplication prevents one entity occupying multiple metric positions.
    ranked = list(dict.fromkeys(ranked))
    positive = {eid for eid, rel in judgments.items() if rel > 0}
    if not positive:
        raise ValueError("Recall/MRR/nDCG require at least one relevant judgment")
    metrics = {}
    for k in ks:
        top = ranked[:k]
        # Binary gains count known positives; judged negatives and unknowns both contribute zero.
        hits = [int(eid in positive) for eid in top]
        # Zero-based i corresponds to one-based rank i+1, hence log2(i+2).
        dcg = sum(hit / np.log2(i + 2) for i, hit in enumerate(hits))
        # The ideal list uses every judged positive up to k, even if never retrieved.
        ideal = sum(1 / np.log2(i + 2) for i in range(min(k, len(positive))))
        metrics[f"recall@{k}"] = sum(hits) / len(positive)
        metrics[f"mrr@{k}"] = next((1 / (i + 1) for i, hit in enumerate(hits) if hit), 0.0)
        metrics[f"ndcg@{k}"] = float(dcg / ideal)
        metrics[f"unjudged@{k}"] = (
            sum(eid not in judgments for eid in top) / len(top) if top else 0.0
        )
        metrics[f"returned@{k}"] = len(top)
    return metrics


def bootstrap(values, samples=1000, seed=650):
    """Estimate a mean and percentile 95% interval by resampling query values.

    Args:
        values: One metric or paired metric difference per evaluated query.
        samples: Number of bootstrap resamples; zero returns only the mean.
        seed: Reproducible NumPy random seed.

    Returns:
        Mapping with mean, ci95 bounds (or None), and sample count n.

    Empty input returns mean=None. This is a query-level descriptive bootstrap,
    not a disease-cluster bootstrap; shared pairs across directions are correlated.
    """
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {"mean": None, "ci95": None, "n": 0}
    mean = float(np.mean(values))
    if not samples:
        return {"mean": mean, "ci95": None, "n": len(values)}
    # Use a local generator so evaluation does not change unrelated global random state.
    rng = np.random.default_rng(seed)
    means = np.array(
        [np.mean(rng.choice(values, len(values), replace=True)) for _ in range(samples)]
    )
    return {"mean": mean, "ci95": np.quantile(means, [0.025, 0.975]).tolist(), "n": len(values)}


def pair_judgments(rows, settings):
    """Select usable pair labels and count temporal/conflict exclusions.

    Temporal mode requires outcome and prior-indication dates, a named prior
    indication, outcome after cutoff, and prior date no later than cutoff.
    A pair with both relevance values is excluded instead of resolved by source
    order. Checking a prior indication against the target name happens later
    when evaluate has access to the entity registry.

    Returns:
        (labels, exclusions), where labels maps (drug_id, disease_id) to 0/1.
    """
    groups = defaultdict(list)
    excluded = defaultdict(int)
    temporal = settings.evaluation.mode == "temporal-repurposing"
    cutoff = (
        date(settings.evaluation.cutoff_year, 12, 31) if settings.evaluation.cutoff_year else None
    )
    for row in rows:
        # Missing temporal facts exclude a label; dataset download dates are never substituted.
        if temporal:
            if not row.confirmed_date or not row.prior_indication_date or not row.prior_indication:
                excluded["missing_verified_temporal_fields"] += 1
                continue
            if row.confirmed_date <= cutoff or row.prior_indication_date > cutoff:
                excluded["outside_temporal_split"] += 1
                continue
        groups[(row.drug_id, row.disease_id)].append(row)
    labels = {}
    # Collapse repeated source rows only when they agree on binary relevance.
    for pair, evidence in groups.items():
        values = {row.relevance for row in evidence}
        if len(values) != 1:
            excluded["conflicting_pair_outcomes"] += 1
        else:
            labels[pair] = values.pop()
    return labels, dict(excluded)


def evaluate(settings, methods=None, directions=None, output: Path | None = None, max_queries=None):
    """Run a fixed entity-query benchmark and persist inspectable results.

    Args:
        settings: Corpus/index, scoring, cutoff, and metric settings.
        methods: Method variants; append +feasibility to enable BBB reranking.
        directions: Supported entity search directions, or None for both.
        output: New run directory; None creates a UTC-timestamped directory.
        max_queries: Optional stable-sorted prefix for smoke tests, not a sample.

    Returns:
        Path to the completed run directory.

    Queries without positives are excluded. Each remaining query records all
    judged positives and separate eligibility/corpus/co-occurrence coverage.
    Every method uses the same selected queries and judgments. Aggregates are
    macro means by direction and jointly; paired deltas compare against BM25
    when it is included. BBB rate summaries apply only to CNS disease queries.

    Outputs include summary/config JSON, per-query CSV, complete ranked results,
    a query manifest, a short README, and the runs/LATEST pointer. Existing run
    summaries are not overwritten. Neural methods may load/download their models.

    Raises:
        ValueError: Methods/directions, labels, dates, index, or run destination
            are unusable for the requested evaluation.
    """
    if output is not None and (output / "summary.json").exists():
        raise ValueError(f"Evaluation output already exists: {output}; choose a new directory")
    # Validate method names before an expensive model-backed benchmark starts.
    methods = methods or ["random", "bm25", "bm25+feasibility", "abc"]
    allowed = {"random", "bm25", "dense", "hybrid", "cross-encoder", "abc"}
    for variant in methods:
        if variant.removesuffix("+feasibility") not in allowed:
            raise ValueError(f"Unknown evaluation method: {variant}")
    if len(set(methods)) != len(methods):
        raise ValueError("Duplicate evaluation methods")
    directions = directions or ["disease-to-drug", "drug-to-disease"]
    if set(directions) - {"disease-to-drug", "drug-to-disease"}:
        raise ValueError("Evaluation supports disease-to-drug and drug-to-disease")
    engine = SearchEngine(settings)
    rows = [
        Judgment.model_validate(r) for r in read_jsonl(settings.paths.processed / "judgments.jsonl")
    ]
    labels, exclusions = pair_judgments(rows, settings)
    if not labels:
        raise ValueError(
            "No usable judgments. Temporal evaluation requires verified dated outcomes; "
            "repoDB alone cannot supply them. Add local/judgments.jsonl or use indication-recovery."
        )
    # One pair supplies a disease query label and a reverse drug query label.
    queries = defaultdict(dict)
    for (drug, disease), rel in labels.items():
        queries[("disease-to-drug", disease)][drug] = rel
        queries[("drug-to-disease", drug)][disease] = rel
    selected = [
        (key, values)
        for key, values in sorted(queries.items())
        if key[0] in directions and any(values.values())
    ]
    # A stable prefix is for smoke testing only; it is not a representative random sample.
    if max_queries:
        selected = selected[:max_queries]
    if not selected:
        raise ValueError("No queries with positive judgments remain")
    query_manifest = []
    for (direction, query_id), judgments in selected:
        query_entity = engine.registry.entities[query_id]
        if (
            settings.evaluation.cutoff_year
            and query_entity.kind == "drug"
            and not engine.candidate_allowed(query_entity)
        ):
            exclusions["query_drug_not_approved_as_of_cutoff"] = (
                exclusions.get("query_drug_not_approved_as_of_cutoff", 0) + 1
            )
            continue
        if settings.evaluation.mode == "temporal-repurposing":
            # A claimed prior different indication must not merely repeat the target disease.
            for row in rows:
                if (row.drug_id, row.disease_id) in labels and row.prior_indication:
                    prior = engine.registry.resolve(row.prior_indication, "disease")
                    if prior and prior.id == row.disease_id:
                        raise ValueError(
                            "Temporal judgment prior_indication equals its target disease"
                        )
        # Preserve all scoped positives, then report eligibility/coverage as separate diagnostics.
        positives = {i for i, rel in judgments.items() if rel}
        eligible = positives & engine.candidate_ids(
            "drug" if direction == "disease-to-drug" else "disease"
        )
        covered = {i for i in positives if engine._entity_docs[i]}
        direct = {i for i in positives if engine._entity_docs[i] & engine._entity_docs[query_id]}
        query_manifest.append(
            {
                "direction": direction,
                "query_id": query_id,
                "query": query_entity.name,
                "judgments": judgments,
                "relevant_total": len(positives),
                "relevant_eligible": len(eligible),
                "relevant_in_corpus": len(covered),
                "relevant_direct_evidence": len(direct),
            }
        )
    if not query_manifest:
        raise ValueError("No eligible dated queries remain")
    # All variants below share the exact same query manifest and judgment denominator.
    per_query, rankings = [], []
    for variant in methods:
        method = variant.removesuffix("+feasibility")
        feasibility = variant.endswith("+feasibility")
        for q in query_manifest:
            response = engine.search(
                q["query_id"],
                q["direction"],
                method,
                min(max(settings.evaluation.ks), 100),
                feasibility,
            )
            ranked = [result.entity.id for result in response.results]
            # Evaluate final entity IDs, not paper IDs or raw model scores.
            metrics = ranking_metrics(ranked, q["judgments"], settings.evaluation.ks)
            entity = engine.registry.entities[q["query_id"]]
            if q["direction"] == "disease-to-drug" and entity.cns:
                for k in settings.evaluation.ks:
                    top = response.results[:k]
                    metrics[f"non_bbb@{k}"] = (
                        (sum(r.entity.bbb == "negative" for r in top) / len(top)) if top else 0.0
                    )
                    metrics[f"unknown_bbb@{k}"] = (
                        (sum(r.entity.bbb in (None, "conflicting") for r in top) / len(top))
                        if top
                        else 0.0
                    )
            per_query.append(
                {
                    "method": variant,
                    **{key: value for key, value in q.items() if key != "judgments"},
                    **metrics,
                }
            )
            rankings.append(
                {**response.model_dump(mode="json"), "method": variant, "query_id": q["query_id"]}
            )
        print(f"Evaluated {variant}: {len(query_manifest)} queries", flush=True)
    # Some diagnostics apply only to CNS queries, so discover the union of recorded metrics.
    metric_keys = sorted({key for row in per_query for key in row if "@" in key})
    # Macro means give equal weight to queries, not to individual relevant pairs.
    summary = {}
    for variant in methods:
        summary[variant] = {}
        for direction in [*directions, "both"]:
            subset = [
                r
                for r in per_query
                if r["method"] == variant and (direction == "both" or r["direction"] == direction)
            ]
            summary[variant][direction] = {
                key: bootstrap(
                    [r[key] for r in subset if key in r],
                    settings.evaluation.bootstrap_samples,
                    settings.retrieval.seed,
                )
                for key in metric_keys
            }
    # Paired differences compare methods on the same query before bootstrapping.
    paired = {}
    if "bm25" in methods:
        baseline = {(r["direction"], r["query_id"]): r for r in per_query if r["method"] == "bm25"}
        for variant in methods:
            if variant == "bm25":
                continue
            subset = [r for r in per_query if r["method"] == variant]
            paired[variant + " minus bm25"] = {
                key: bootstrap(
                    [
                        r[key] - baseline[(r["direction"], r["query_id"])][key]
                        for r in subset
                        if key in r
                    ],
                    settings.evaluation.bootstrap_samples,
                    settings.retrieval.seed,
                )
                for key in metric_keys
            }
    # Use a new UTC-named run folder by default; never overwrite an earlier summary.
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    output = output or settings.paths.runs / stamp
    if (output / "summary.json").exists():
        raise ValueError(f"Evaluation output already exists: {output}; choose a new directory")
    output.mkdir(parents=True, exist_ok=True)
    write_json(
        output / "summary.json",
        {
            "mode": settings.evaluation.mode,
            "cutoff_year": settings.evaluation.cutoff_year,
            "query_count": len(query_manifest),
            "exclusions": exclusions,
            "metrics": summary,
            "paired_deltas": paired,
            "index_hash": engine.manifest["index_hash"],
            "judgments_sha256": digest(settings.paths.processed / "judgments.jsonl"),
            "model_revisions": engine.model_revisions,
            "packages": package_versions(),
            "limitations": [
                "Unjudged pairs score zero; unjudged@k measures their share of returned results.",
                "Indication recovery is not evidence of novel repurposing or clinical efficacy.",
                "Time-filtered literature does not date pretrained models or current BBB labels.",
            ],
        },
    )
    # Save the resolved configuration beside metrics so an experiment can be reconstructed.
    write_json(output / "config.json", settings.model_dump(mode="json"))
    write_jsonl(output / "queries.jsonl", query_manifest)
    write_jsonl(output / "rankings.jsonl", rankings)
    fields = sorted({key for row in per_query for key in row})
    with (output / "per-query.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(per_query)
    lines = [
        "# Evaluation run",
        "",
        f"Mode: {settings.evaluation.mode}",
        f"Queries: {len(query_manifest)}; all metrics are macro averages over queries.",
        "",
        "| Method | Recall@10 | MRR@10 | nDCG@10 |",
        "|---|---:|---:|---:|",
    ]
    for variant in methods:
        data = summary[variant]["both"]
        cells = [
            f"{data[key]['mean']:.4f}" if key in data else "—"
            for key in ["recall@10", "mrr@10", "ndcg@10"]
        ]
        lines.append("| " + " | ".join([variant, *cells]) + " |")
    lines += [
        "",
        "These starter-corpus results measure known indication recovery. "
        "See summary.json for confidence intervals, exclusions, and limitations.",
    ]
    atomic_text(output / "README.md", "\n".join(lines) + "\n")
    # LATEST is a text pointer, not a directory containing another copy of the run.
    atomic_text(settings.paths.runs / "LATEST", str(output.resolve()) + "\n")
    return output


def package_versions():
    """Record installed versions of key runtime packages for run provenance.

    Missing optional packages are omitted rather than preventing lexical-only
    evaluation. This records versions; it does not install or modify packages.
    """
    result = {}
    for package in ["drug-repurpose-ir", "numpy", "rank-bm25", "sentence-transformers", "torch"]:
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            pass
    return result
