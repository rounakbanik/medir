"""Behavioral tests for retrieval, evaluation, date handling, and the optional API.

Neural ranking uses a fixed two-dimensional encoder fixture rather than
downloading models. Real-model smoke verification is recorded separately
in docs/v1-validation.md; these tests should stay fast and offline.
"""

import json
from datetime import date
from xml.etree import ElementTree as ET

import numpy as np
import pytest

from drug_repurpose_ir.engine import SearchEngine, rrf
from drug_repurpose_ir.entities import Registry
from drug_repurpose_ir.evaluation import bootstrap, evaluate, pair_judgments, ranking_metrics
from drug_repurpose_ir.indexing import build_index, dense_signature
from drug_repurpose_ir.io import read_jsonl, write_json, write_jsonl
from drug_repurpose_ir.models import Entity, Judgment
from drug_repurpose_ir.pubmed import parse_date, read_pubmed


def test_metrics_keep_missing_relevant_in_denominator():
    """Check hand-computable metric values and stable duplicate removal.

    The missing positive "absent" must lower recall even when the ranker never
    returns it. Large cutoffs and empty result lists must remain well-defined.
    """
    values = ranking_metrics(["a", "x", "a", "b"], {"a": 1, "b": 0, "absent": 1}, [1, 3, 10])
    assert values["recall@1"] == 0.5
    assert values["recall@10"] == 0.5
    assert values["mrr@3"] == 1
    assert values["unjudged@3"] == pytest.approx(1 / 3)
    assert values["ndcg@3"] == pytest.approx(1 / (1 + 1 / np.log2(3)))
    assert values["returned@10"] == 3
    assert ranking_metrics([], {"a": 1}, [10])["recall@10"] == 0


def test_ambiguous_names_and_token_boundaries():
    """Ensure shared aliases remain unresolved and substrings do not become mentions."""
    registry = Registry(
        [
            Entity(id="a", kind="drug", name="Alpha", aliases=["common"]),
            Entity(id="b", kind="drug", name="Beta", aliases=["common"]),
        ]
    )
    assert registry.resolve("common") is None
    assert registry.mentions("alphabet common BETA") == {"b"}
    assert registry.mentions("Alpha and Beta") == {"a", "b"}


def test_bidirectional_evidence_approval_and_bbb(corpus):
    """Exercise both query directions, evidence references, and all BBB adjustment cases.

    Approved candidates with supporting papers are retained, unapproved or
    unsupported ones are excluded, and unknown query names fail explicitly.
    """
    engine = SearchEngine(corpus)
    response = engine.search("Brain disease", k=10)
    ids = [r.entity.id for r in response.results]
    assert "drug:alpha" in ids
    assert "drug:unapproved" not in ids
    assert "drug:absent" not in ids
    assert all(r.evidence for r in response.results)
    reverse = engine.search("Alpha medicine", direction="drug-to-disease")
    assert "disease:brain" in {r.entity.id for r in reverse.results}
    reranked = engine.search("Brain disease", k=10, feasibility=True)
    adjustments = {r.entity.id: r.feasibility_adjustment for r in reranked.results}
    assert adjustments["drug:alpha"] == 0.15
    assert adjustments["drug:beta"] == -0.15
    assert adjustments["drug:unknown"] == 0
    with pytest.raises(ValueError, match="unknown or ambiguous"):
        engine.search("made-up disease")


def test_random_reproducible_and_includes_uncovered_entities(corpus):
    """Require deterministic random ordering over the full eligible catalogue.

    Entities with no papers must still be eligible for the random baseline so
    its candidate universe is not secretly reduced to easy covered entities.
    """
    engine = SearchEngine(corpus)
    a = engine.search("Brain disease", method="random", k=100)
    b = engine.search("Brain disease", method="random", k=100)
    assert a == b
    assert "drug:absent" in {r.entity.id for r in a.results}


def test_rrf_and_dense_alignment_without_network(corpus, monkeypatch):
    """Validate RRF arithmetic and dense row mapping using a fake encoder/matrix.

    The fixed query vector selects a known document row. Changing the requested
    model must invalidate the saved dense metadata instead of reusing its rows.
    """
    assert rrf([[1, 2], [2, 3]], 60)[2] == pytest.approx(1 / 62 + 1 / 61)
    engine = SearchEngine(corpus)
    n = len(engine.documents)
    # Give only row zero a nonzero score for the fixed fake query vector.
    matrix = np.zeros((n, 2), dtype=np.float32)
    matrix[0, 0] = 1
    np.save(corpus.paths.index / "embeddings.npy", matrix)
    write_json(
        corpus.paths.index / "dense.json",
        {"signature": dense_signature(corpus, engine.manifest["index_hash"]), "shape": [n, 2]},
    )

    class FakeEncoder:
        """Minimal SentenceTransformer substitute returning a controlled query vector."""

        def encode(self, text, **kwargs):
            """Ignore natural-language inputs and return the vector used by this test matrix."""
            return np.array([1, 0], dtype=np.float32)

    # Replace the model loader only, leaving production retrieval and row alignment intact.
    monkeypatch.setattr("drug_repurpose_ir.engine.encoder", lambda _, **kw: FakeEncoder())
    response = engine.search("Brain disease", method="dense")
    assert response.results[0].entity.id == "drug:alpha"
    assert engine.search("Brain disease", method="hybrid").results
    corpus.retrieval.dense_model = "different"
    with pytest.raises(ValueError, match="stale"):
        SearchEngine(corpus).search("Brain disease", method="dense")


def test_stale_index_and_temporal_corpus(corpus):
    """Verify cutoff changes invalidate snapshots and exclude later/undated approvals."""
    corpus.evaluation.cutoff_year = 2011
    with pytest.raises(ValueError, match="stale"):
        SearchEngine(corpus)
    result = build_index(corpus)
    assert result["excluded_by_date"] == 3
    engine = SearchEngine(corpus)
    assert all(d.publication_date.year <= 2011 for d in engine.documents)
    assert "drug:absent" not in engine.candidate_ids("drug")  # Approval date is unknown.


def test_temporal_labels_require_dates_and_different_indication(corpus):
    """Reject undated temporal labels, conflicting outcomes, and repeated target indications.

    A reviewed date example is admitted only after the cutoff with an earlier
    prior indication; ordinary repoDB-style undated records must not fabricate
    a historical repurposing benchmark.
    """
    corpus.evaluation.cutoff_year = 2015
    corpus.evaluation.mode = "temporal-repurposing"
    rows = [
        Judgment.model_validate(row)
        for row in read_jsonl(corpus.paths.processed / "judgments.jsonl")
    ]
    labels, excluded = pair_judgments(rows, corpus)
    assert labels == {}
    assert excluded["missing_verified_temporal_fields"] == len(rows)
    dated = rows[0].model_copy(
        update={
            "confirmed_date": date(2016, 1, 1),
            "prior_indication_date": date(2010, 1, 1),
            "prior_indication": "Viral infection",
        }
    )
    labels, _ = pair_judgments([dated], corpus)
    assert labels[("drug:alpha", "disease:brain")] == 1
    negative = dated.model_copy(update={"relevance": 0})
    assert pair_judgments([dated, negative], corpus)[0] == {}
    build_index(corpus)
    with pytest.raises(ValueError, match="No usable judgments"):
        evaluate(corpus)
    dated.prior_indication = "Brain disease"
    write_jsonl(corpus.paths.processed / "judgments.jsonl", [dated])
    with pytest.raises(ValueError, match="equals its target"):
        evaluate(corpus)


def test_evaluation_outputs_and_fixed_coverage(corpus):
    """Check saved query coverage, preserved method variants, metrics, and bootstrap seeds."""
    output = evaluate(corpus, ["bm25", "random", "bm25+feasibility"])
    summary = json.loads((output / "summary.json").read_text())
    assert summary["query_count"] == 4
    queries = list(read_jsonl(output / "queries.jsonl"))
    brain = next(q for q in queries if q["query_id"] == "disease:brain")
    # The corpus-absent positive stays in the denominator instead of making recall look better.
    assert brain["relevant_total"] == 2
    assert brain["relevant_in_corpus"] == 1
    assert set(r["method"] for r in read_jsonl(output / "rankings.jsonl")) == {
        "bm25",
        "random",
        "bm25+feasibility",
    }
    assert summary["metrics"]["bm25"]["disease-to-drug"]["recall@10"]["mean"] == 0.75
    assert (output / "per-query.csv").exists()
    assert bootstrap([0.2, 0.8], 50, 650) == bootstrap([0.2, 0.8], 50, 650)


def test_pubmed_mixed_content_dates_and_ids(tmp_path):
    """Preserve nested XML text/source IDs and conservative month/year date semantics."""
    path = tmp_path / "sample.xml"
    path.write_text("""<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>123</PMID>
      <Article><ArticleTitle>A <i>trial</i></ArticleTitle>
      <Journal><JournalIssue><PubDate><Year>2010</Year></PubDate></JournalIssue></Journal>
      <Abstract><AbstractText Label="RESULTS">Alpha <b>improves</b> outcomes.</AbstractText>
      </Abstract>
      </Article><MeshHeadingList><MeshHeading><DescriptorName UI="D1">Brain disease</DescriptorName>
      </MeshHeading></MeshHeadingList></MedlineCitation></PubmedArticle></PubmedArticleSet>""")
    doc = next(read_pubmed(path))
    assert doc.title == "A trial"
    assert doc.abstract == "RESULTS: Alpha improves outcomes."
    assert doc.publication_date == date(2010, 12, 31)
    assert doc.date_precision == "year"
    assert doc.mesh_ids == {"Brain disease": "D1"}
    assert parse_date(ET.fromstring("<PubDate><MedlineDate>1998 Winter</MedlineDate></PubDate>"))[
        0
    ] == date(1998, 12, 31)


def test_api_validation_and_search(corpus):
    """Exercise health/search/catalogue routes and reject invalid HTTP request values.

    Skip this test when the optional FastAPI dependency is absent; lexical
    installations should still be able to run the other offline tests.
    """
    # API dependencies are optional; the rest of the offline suite still runs without them.
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from drug_repurpose_ir.api import create_app

    client = TestClient(create_app(corpus))
    assert client.get("/health").json()["documents"] == 7
    response = client.post("/search", json={"query": "Brain disease", "k": 3})
    assert response.status_code == 200
    assert response.json()["results"]
    assert client.post("/search", json={"query": "  "}).status_code == 422
    assert (
        client.post("/search", json={"query": "Brain disease", "method": "typo"}).status_code == 422
    )
    assert client.get("/entities?kind=disease").status_code == 200
