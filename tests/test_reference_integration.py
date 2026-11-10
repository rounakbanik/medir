"""Check the copied coursework and its connections to the biomedical pipeline."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from drug_repurpose_ir.engine import SearchEngine
from drug_repurpose_ir.evaluation import ranking_metrics
from drug_repurpose_ir.indexing import build_index, signature
from drug_repurpose_ir.ir import relevance
from drug_repurpose_ir.ir.indexing import BasicInvertedIndex, PositionalInvertedIndex
from drug_repurpose_ir.ir.ranker import BM25, Ranker
from drug_repurpose_ir.reference_adapter import (
    LEXICAL_METHODS,
    lexical_query,
    reference_cross_encoder,
    reference_metrics,
    reference_ranker,
)


def test_reference_source_preserved():
    """Only the import replacements recorded in provenance.json may differ."""
    root = Path(__file__).resolve().parents[1]
    packaged = root / "src/drug_repurpose_ir/ir"
    manifest = json.loads((packaged / "provenance.json").read_text())
    for filename, info in manifest["files"].items():
        actual = (packaged / filename).read_bytes()
        assert hashlib.sha256(actual).hexdigest() == info["packaged_sha256"]
        original_path = root / "reference/starter-code" / filename
        # Runtime/package use does not require the ignored reference directory on a fresh clone.
        if original_path.exists():
            original = original_path.read_bytes()
            assert hashlib.sha256(original).hexdigest() == info["source_sha256"]
            expected = original
            for before, after in info["import_changes"].items():
                expected = expected.replace(before.encode(), after.encode())
            assert actual == expected


def test_explicit_numeric_defaults_do_not_invalidate_index(corpus):
    """YAML float coercion must not make the reference k3=8 default look different."""
    assert signature({"k3": 8}) == signature({"k3": 8.0})
    corpus.retrieval.bm25_k3 = 8.0
    assert SearchEngine(corpus).ranker.scorer.k3 == 8


def test_export_index_and_ranker_use_reference_classes(corpus):
    """Integer rows adapt PubMed/local IDs without replacing index/ranker logic."""
    engine = SearchEngine(corpus)
    assert type(engine.document_index) is BasicInvertedIndex
    assert type(engine.ranker) is Ranker
    assert type(engine.ranker.scorer) is BM25
    assert engine.ranker.scorer.k1 == 1.2
    rows = [
        json.loads(line) for line in (corpus.paths.index / "corpus.jsonl").read_text().splitlines()
    ]
    assert [row["docid"] for row in rows] == list(range(len(engine.documents)))
    for method in LEXICAL_METHODS:
        direct = reference_ranker(engine.document_index, corpus, method)
        expected = lexical_query(direct, "Brain disease", method)
        assert engine.retrieve("Brain disease", method) == expected


def test_negative_reference_bm25_scores_remain_ranked(corpus):
    """Common terms have negative IDF in the coursework formula; retain those hits."""
    engine = SearchEngine(corpus)
    ranked = engine.retrieve("Brain disease", "bm25")
    assert ranked and all(score < 0 for _, score in ranked)
    response = engine.search("Brain disease", k=10)
    assert response.results
    assert len({result.retrieval_score for result in response.results}) > 1
    assert all(result.evidence[0].score < 0 for result in response.results)


@pytest.mark.parametrize("method", ["tf-idf", "dirichlet"])
def test_unseen_query_terms_are_adapted_before_scoring(corpus, method):
    """The original scorers need nonzero document/collection frequencies."""
    engine = SearchEngine(corpus)
    expected = engine.retrieve("Brain disease", method)
    assert engine.retrieve("Brain disease zzzunknownterm", method) == expected
    assert engine.retrieve("zzzunknownterm", method) == []


def test_positional_reference_index_roundtrip(corpus):
    """The same adapter can use the supplied positional index and persistence."""
    corpus.retrieval.index_type = "positional"
    build_index(corpus)
    engine = SearchEngine(corpus)
    assert type(engine.document_index) is PositionalInvertedIndex
    postings = engine.document_index.get_postings("brain")
    assert postings and len(postings[0]) == 3
    assert engine.search("Brain disease").results


def test_original_metrics_are_called_and_short_lists_are_padded(monkeypatch):
    """Adapt array lengths, keeping both supplied metric formulas intact."""
    calls = []
    original_map, original_ndcg = relevance.map_score, relevance.ndcg_score

    def record_map(values, cut_off):
        calls.append(("map", list(values), cut_off))
        return original_map(values, cut_off)

    def record_ndcg(values, ideal, cut_off):
        calls.append(("ndcg", list(values), list(ideal), cut_off))
        return original_ndcg(values, ideal, cut_off)

    monkeypatch.setattr(relevance, "map_score", record_map)
    monkeypatch.setattr(relevance, "ndcg_score", record_ndcg)
    map_value, ndcg_value = reference_metrics([1], [1, 1], 3)
    assert map_value == 1
    assert ndcg_value == pytest.approx(1 / (1 + 1 / np.log2(3)))
    assert calls == [("map", [1], 3), ("ndcg", [1, 0, 0], [1, 1, 0], 3)]
    values = ranking_metrics(["a"], {"a": 1, "absent": 1}, [3])
    assert values["recall@3"] == 0.5
    assert values["map@3"] == 1  # Preserve the coursework denominator rather than redefining MAP.


def test_reference_cross_encoder_score_interface(corpus, monkeypatch):
    """The scorer gets integer IDs and raw text, and its original score() is used."""
    calls = []

    class FakeModel:
        """Minimal CrossEncoder substitute for checking adapter inputs."""

        model = None

        def __init__(self, name):
            self.model = self

        def to(self, device):
            return self

        def predict(self, pairs):
            calls.extend(pairs)
            return np.array([2.5])

    monkeypatch.setattr("drug_repurpose_ir.ir.ranker.CrossEncoder", FakeModel)
    engine = SearchEngine(corpus)
    scorer = reference_cross_encoder(corpus, engine.documents)
    assert scorer.score(0, "Brain disease") == 2.5
    assert calls == [("Brain disease", " ".join(engine.documents[0].text.split()[:500]))]
