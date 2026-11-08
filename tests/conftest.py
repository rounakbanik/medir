"""Shared synthetic corpus used by offline, deterministic regression tests.

All entities, dates, and evidence here are fictional. Temporary directories
isolate tests from downloaded biomedical data and from each other's outputs.
"""

from datetime import date

import pytest

from drug_repurpose_ir.config import Paths, Settings
from drug_repurpose_ir.indexing import build_index
from drug_repurpose_ir.io import write_jsonl
from drug_repurpose_ir.models import Document, Entity, Judgment


@pytest.fixture
def corpus(tmp_path):
    """Build a tiny indexed corpus exercising approval, BBB, coverage, and dates.

    Alpha is BBB-positive, Beta is negative, and Unknown has no BBB label.
    Unapproved must be filtered; Absent is a judged positive with no paper and
    must remain in recall denominators. Control papers keep BM25 IDF meaningful.
    Returns Settings pointing only to pytest-managed temporary directories.
    """
    # Every test receives its own temporary filesystem; real source data is never touched.
    settings = Settings(
        paths=Paths(
            raw=tmp_path / "raw",
            processed=tmp_path / "processed",
            index=tmp_path / "index",
            runs=tmp_path / "runs",
        )
    )
    settings.retrieval.candidate_documents = 100
    settings.evaluation.ks = [1, 3, 10]
    settings.evaluation.bootstrap_samples = 30
    # Include positive, negative, unknown, unapproved, and corpus-absent cases explicitly.
    entities = [
        Entity(id="disease:brain", kind="disease", name="Brain disease", cns=True),
        Entity(id="disease:infection", kind="disease", name="Viral infection"),
        Entity(
            id="drug:alpha",
            kind="drug",
            name="Alpha",
            aliases=["Alpha medicine"],
            approved=True,
            approved_since=date(2000, 1, 1),
            bbb="positive",
        ),
        Entity(
            id="drug:beta",
            kind="drug",
            name="Beta",
            approved=True,
            approved_since=date(2001, 1, 1),
            bbb="negative",
        ),
        Entity(
            id="drug:unknown",
            kind="drug",
            name="Unknown",
            approved=True,
            approved_since=date(2002, 1, 1),
        ),
        Entity(id="drug:unapproved", kind="drug", name="Unapproved"),
        Entity(id="drug:absent", kind="drug", name="Absent", approved=True),
    ]
    # The different years let cutoff tests admit/exclude known rows deterministically.
    documents = [
        Document(
            id="p1",
            title="Brain disease trial",
            abstract="Alpha treats brain disease.",
            publication_date=date(2010, 1, 1),
            entity_ids=["disease:brain", "drug:alpha"],
            mesh=["Kinase"],
        ),
        Document(
            id="p2",
            abstract="Beta is studied in brain disease for possible treatment.",
            publication_date=date(2011, 1, 1),
            entity_ids=["disease:brain", "drug:beta"],
            mesh=["Protein"],
        ),
        Document(
            id="p3",
            abstract="Unknown with brain disease.",
            publication_date=date(2012, 1, 1),
            entity_ids=["disease:brain", "drug:unknown"],
        ),
        Document(
            id="p4",
            abstract="Unapproved treats brain disease.",
            publication_date=date(2013, 1, 1),
            entity_ids=["disease:brain", "drug:unapproved"],
        ),
        Document(
            id="p5",
            abstract="Alpha viral infection.",
            publication_date=date(2020, 1, 1),
            entity_ids=["disease:infection", "drug:alpha"],
            mesh=["Kinase"],
        ),
        Document(
            id="p6",
            abstract="A control paper with unrelated science and laboratory data.",
            publication_date=date(2010, 1, 1),
        ),
        Document(
            id="p7",
            abstract="Another control on different matters without linked entities.",
            publication_date=date(2010, 1, 1),
        ),
    ]
    # Absent is deliberately relevant even though no document mentions it.
    judgments = [
        Judgment(drug_id="drug:alpha", disease_id="disease:brain", relevance=1, source="test"),
        Judgment(drug_id="drug:absent", disease_id="disease:brain", relevance=1, source="test"),
        Judgment(drug_id="drug:beta", disease_id="disease:brain", relevance=0, source="test"),
        Judgment(drug_id="drug:alpha", disease_id="disease:infection", relevance=1, source="test"),
    ]
    write_jsonl(settings.paths.processed / "entities.jsonl", entities)
    write_jsonl(settings.paths.processed / "documents.jsonl", documents)
    write_jsonl(settings.paths.processed / "judgments.jsonl", judgments)
    # Tests exercise the actual snapshot loader, not a separate in-memory toy implementation.
    build_index(settings)
    return settings
