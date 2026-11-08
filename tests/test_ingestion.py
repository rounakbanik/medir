"""Offline tests for source adaptation, provenance, and input failure handling."""

import json
from datetime import date

import httpx
import pytest

from drug_repurpose_ir.acquisition import Downloader
from drug_repurpose_ir.config import Paths, Settings
from drug_repurpose_ir.ingestion import ingest
from drug_repurpose_ir.io import read_jsonl, write_json, write_jsonl
from drug_repurpose_ir.models import Document, Entity


def test_source_ingestion_dedup_labels_and_linking(tmp_path):
    """Verify the full raw-to-canonical join using tiny fictional source exports.

    Checks that a brand links to its ingredient, duplicated text collapses,
    external IDs survive, and repoDB labels do not grant independent FDA approval
    or leak status words into the searchable document text.
    """
    s = Settings(paths=Paths(raw=tmp_path / "raw", processed=tmp_path / "processed"))
    s.diseases = [Entity(id="disease:brain", kind="disease", name="Brain disease", cns=True)]
    # Arrange: create tiny fictional source exports in the real expected folder layout.
    fda = s.paths.raw / "openfda" / "drugs.json"
    write_json(
        fda,
        {
            "results": [
                {
                    "application_number": "NDA1",
                    "submissions": [
                        {
                            "submission_type": "ORIG",
                            "submission_status": "AP",
                            "submission_status_date": "20000101",
                        }
                    ],
                    "products": [
                        {"brand_name": "AlphaBrand", "active_ingredients": [{"name": "Alpha"}]}
                    ],
                }
            ]
        },
    )
    repodb = s.paths.raw / "repodb" / "full.csv"
    repodb.parent.mkdir(parents=True)
    repodb.write_text(
        "drug_name,drug_id,ind_name,ind_id,status\n"
        "Alpha,DB1,Brain disease,C1,Approved\n"
        "Beta,DB2,Brain disease,C1,Terminated\n"
    )
    bbb = s.paths.raw / "b3db" / "classification.tsv"
    bbb.parent.mkdir(parents=True)
    bbb.write_text("compound_name\tBBB+/BBB-\tCID\nAlpha\tBBB+\t1\n")
    docs = [
        Document(
            id="p1",
            title="Trial",
            abstract="AlphaBrand for Brain disease.",
            mesh_ids={"Brain disease": "D1"},
            publication_date=date(2010, 1, 1),
        ),
        Document(id="p2", title="Trial", abstract="AlphaBrand for Brain disease."),
    ]
    write_jsonl(s.paths.raw / "local" / "documents.jsonl", docs)
    # Act: run the production adapter/linking/deduplication pipeline on those exports.
    report = ingest(s)
    # Assert: duplicated text collapses even though its two source IDs differ.
    assert report["documents"] == 1
    assert report["counts"]["duplicate_documents"] == 1
    entities = {row["id"]: row for row in read_jsonl(s.paths.processed / "entities.jsonl")}
    assert entities["drug:alpha"]["approved"] is True
    assert entities["drug:alpha"]["bbb"] == "positive"
    assert entities["drug:beta"]["approved"] is None  # Labels do not confer approval.
    assert entities["disease:brain"]["identifiers"] == {"umls": ["C1"], "mesh": ["D1"]}
    actual_doc = next(read_jsonl(s.paths.processed / "documents.jsonl"))
    assert actual_doc["entity_ids"] == ["disease:brain", "drug:alpha"]
    assert "Approved" not in actual_doc["abstract"]
    assert report["judgments"] == 2
    assert ingest(s)["documents"] == 1


def test_downloader_cache_checksum_retry_and_redaction(tmp_path):
    """Verify cache reuse, corrupted-file refetch, and secret-free manifests.

    MockTransport records requests without contacting the network. The second
    identical save must reuse the file; corrupting bytes must trigger refetch.
    The function name mentions retry, but transport/backoff retries are not
    exercised by this particular successful-response fixture.
    """
    calls = []

    def handler(request):
        """Record a fake HTTP request and return a fixed in-memory dataset payload."""
        calls.append(request)
        return httpx.Response(200, content=b"dataset")

    downloader = Downloader(tmp_path)
    downloader.client.close()
    # Intercept HTTP entirely; these tests must work without network access.
    downloader.client = httpx.Client(transport=httpx.MockTransport(handler))
    target = tmp_path / "data.csv"
    downloader.save(target, "https://example.org/data", {"api_key": "secret", "email": "private"})
    downloader.save(target, "https://example.org/data", {"api_key": "secret", "email": "private"})
    assert len(calls) == 1
    manifest = target.with_name("data.csv.manifest.json").read_text()
    assert "secret" not in manifest and "private" not in manifest
    assert json.loads(manifest)["sha256"]
    # A file still exists, but its checksum must stop it being trusted as a cache hit.
    target.write_bytes(b"corrupted")
    downloader.save(target, "https://example.org/data", {"api_key": "secret", "email": "private"})
    assert len(calls) == 2
    assert target.read_bytes() == b"dataset"
    downloader.close()


def test_missing_corpus_fails_instead_of_pretending_success(tmp_path):
    """Require an explicit ingestion error when no literature documents are available."""
    settings = Settings(paths=Paths(raw=tmp_path / "raw", processed=tmp_path / "processed"))
    with pytest.raises(ValueError, match="No documents found"):
        ingest(settings)
