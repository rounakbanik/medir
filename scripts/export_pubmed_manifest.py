#!/usr/bin/env python3
"""Export shareable PubMed acquisition metadata without abstract text."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(raw_dir: Path, name: str) -> dict:
    path = raw_dir / name
    sidecar = path.with_name(path.name + ".manifest.json")
    if not sidecar.exists():
        raise FileNotFoundError(f"Missing acquisition manifest for {path}")
    metadata = read_json(sidecar)
    actual_sha256 = sha256(path)
    if metadata["sha256"] != actual_sha256:
        raise ValueError(f"Checksum mismatch for {path}")
    return {
        "file": name,
        "sha256": actual_sha256,
        "bytes": metadata["bytes"],
        "retrieved_at_utc": metadata["retrieved_at"],
    }


def build_manifest(raw_dir: Path, report_path: Path, processed_docs: Path) -> dict:
    active_path = raw_dir / "active-files.json"
    if not active_path.exists():
        raise FileNotFoundError(f"Missing successful PubMed selection file: {active_path}")
    active_files = json.loads(active_path.read_text(encoding="utf-8"))
    query_files = sorted(raw_dir.glob("*.search.json"))
    if not query_files:
        raise FileNotFoundError(f"No PubMed ESearch results found in {raw_dir}")

    queries = []
    retrieved_times = []
    total_selected = 0
    for query_path in query_files:
        key = query_path.name.removesuffix(".search.json")
        search_manifest = read_json(query_path.with_name(query_path.name + ".manifest.json"))
        result = read_json(query_path)["esearchresult"]
        pmids = result.get("idlist", [])
        total_selected += len(pmids)
        search_file = file_record(raw_dir, query_path.name)
        retrieved_times.append(search_manifest["retrieved_at"])
        xml_names = sorted(
            name for name in active_files if name.startswith(f"{key}-") and name.endswith(".xml")
        )
        batch_files = [file_record(raw_dir, name) for name in xml_names]
        retrieved_times.extend(item["retrieved_at_utc"] for item in batch_files)
        queries.append(
            {
                "key": key,
                "query": search_manifest["parameters"]["term"],
                "sort": search_manifest["parameters"].get("sort"),
                "limit": search_manifest["parameters"].get("retmax"),
                "total_hits_at_retrieval": int(result["count"]),
                "selected_count": len(pmids),
                "selected_pmids": pmids,
                "search_result_file": search_file,
                "efetch_files": batch_files,
            }
        )

    retrieved_at_utc = max(retrieved_times)
    snapshot_date = datetime.fromisoformat(retrieved_at_utc).date().isoformat()
    limits = sorted({query["limit"] for query in queries})
    report = read_json(report_path)
    return {
        "schema_version": 1,
        "snapshot_id": f"pubmed-expanded-{snapshot_date}",
        "source": "NCBI PubMed via E-utilities",
        "retrieved_at_utc": retrieved_at_utc,
        "note": "PMID/query/checksum metadata only; abstract text is not included.",
        "baseline": {
            "snapshot_date": "2026-11-08",
            "per_disease_limit": 100,
            "unique_abstracts": 1922,
            "reference": "docs/v1-validation.md",
        },
        "expanded_acquisition": {
            "per_query_limit": limits[0] if len(limits) == 1 else limits,
            "query_count": len(queries),
            "selected_query_pmids_total": total_selected,
            "documents_read_before_deduplication": report["counts"]["documents_read"],
            "duplicate_documents": report["counts"]["duplicate_documents"],
            "unique_abstracts": report["documents"],
            "normalized_documents_sha256": sha256(processed_docs),
            "disease_document_counts": report["disease_document_counts"],
            "queries": queries,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw/pubmed")
    parser.add_argument(
        "--report", type=Path, default=ROOT / "data/processed/ingestion-report.json"
    )
    parser.add_argument(
        "--documents", type=Path, default=ROOT / "data/processed/documents.jsonl"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    manifest = build_manifest(args.raw_dir, args.report, args.documents)
    if args.output is None:
        args.output = ROOT / "data/manifests" / f"{manifest['snapshot_id']}.json"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        "Wrote metadata for "
        f"{manifest['expanded_acquisition']['unique_abstracts']} unique abstracts to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
