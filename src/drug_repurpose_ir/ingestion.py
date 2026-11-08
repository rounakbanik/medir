"""Adapt raw datasets into canonical entities, literature, and separate judgments.

This is the main place to read when adding a data source. Stage order matters:
build the entity catalogue first, enrich approval/BBB metadata, construct
independent labels, then link/deduplicate literature. Generated files live
under the configured processed directory; source files are not rewritten.
"""

import csv
import gzip
import json
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path

from .config import Settings
from .entities import Registry, drug_id, text_hash
from .io import digest, read_jsonl, write_json, write_jsonl
from .models import Document, Entity, Judgment
from .pubmed import read_pubmed


def table_rows(path: Path, delimiter=","):
    """Yield dictionaries from a delimited UTF-8 table, optionally gzip-compressed.

    utf-8-sig tolerates a leading byte-order mark. Delimiter selection belongs to
    the source adapter: comma for CSV, tab for TSV. Source columns remain strings.
    """
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig", newline="") as stream:
        yield from csv.DictReader(stream, delimiter=delimiter)


def fda_rows(path: Path):
    """Yield Drugs@FDA application objects from a JSON export or JSON ZIP.

    Bulk ZIP members are opened directly without extracting paths to disk.
    Each JSON payload is loaded in memory; this v1 is not a streaming JSON parser.
    Plain JSON may be a results envelope or a list of application objects.
    """
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            for name in archive.namelist():
                if name.endswith(".json"):
                    with archive.open(name) as stream:
                        yield from json.load(stream)["results"]
    else:
        value = json.loads(path.read_text())
        yield from value["results"] if isinstance(value, dict) else value


def ingest(settings: Settings):
    """Normalize all configured raw inputs and write the reusable processed dataset.

    Args:
        settings: Raw/processed paths and the configured disease catalogue.

    Returns:
        Coverage/provenance report also saved as ingestion-report.json.

    Outputs:
        entities.jsonl, documents.jsonl, judgments.jsonl, ingestion-report.json.

    Configured/local entities seed the registry. RepoDB supplies vocabulary and
    separately held labels; FDA supplies approval evidence; local aliases and
    B3DB enrich the registry. Documents are validated, deduplicated by ID/DOI/text,
    and linked to canonical IDs. The first duplicate encountered wins.

    Raises:
        ValueError: Required source columns, referenced IDs, or corpus are invalid.
        pydantic.ValidationError: A normalized/local record violates its schema.

    Re-running overwrites derived files deterministically. It does not merge
    arbitrarily named files in the project root or watch for changes in real time.
    """
    # Keep source exports and generated canonical files in different directories.
    raw, output = settings.paths.raw, settings.paths.processed
    # Copy configured entities because source enrichment mutates canonical records.
    registry = Registry(d.model_copy(deep=True) for d in settings.diseases)
    counts = Counter()
    inputs = []

    def files(folder, suffixes):
        """Discover sorted source files by suffix and record them for the input manifest.

        Nested directories are included. Sorting makes first-record-wins duplicate
        handling reproducible for an unchanged set of input files.
        """
        found = sorted(
            p
            for p in (raw / folder).rglob("*")
            if p.is_file() and any(p.name.endswith(s) for s in suffixes)
        )
        inputs.extend(found)
        return found

    # Stage 1: load explicit local entity records before matching external source names.
    for path in files("local", ("entities.jsonl",)):
        for row in read_jsonl(path):
            registry.add(Entity.model_validate(row))
    # repoDB supplies drug NAME/ID vocabulary only. Relationships never enter the index.
    repodb = []
    for path in files("repodb", (".csv", ".tsv")):
        for row in table_rows(path, "\t" if path.suffix == ".tsv" else ","):
            if not {"drug_name", "drug_id", "ind_name", "status"} <= row.keys():
                raise ValueError(
                    f"{path}: expected repoDB drug_name,drug_id,ind_name,status columns"
                )
            repodb.append(row)
            name = row["drug_name"].strip()
            if name:
                existing = registry.resolve(name, "drug")
                registry.add(
                    Entity(
                        id=existing.id if existing else drug_id(name),
                        kind="drug",
                        name=name,
                        identifiers={"drugbank": [row["drug_id"]]} if row["drug_id"] else {},
                        sources=["repodb:vocabulary"],
                    )
                )
    # Stage 2: attach independent FDA approval evidence to single-ingredient entities.
    for path in files("openfda", (".json.zip", ".json")):
        # Downloader manifests share .json suffixes but are metadata, not FDA application exports.
        if path.name.endswith(".manifest.json"):
            continue
        for row in fda_rows(path):
            counts["fda_applications"] += 1
            # Original approved submissions determine the earliest observed availability date.
            dates = []
            for sub in row.get("submissions", []):
                if sub.get("submission_status") == "AP" and sub.get("submission_type") == "ORIG":
                    try:
                        dates.append(
                            datetime.strptime(sub["submission_status_date"], "%Y%m%d").date()
                        )
                    except (KeyError, ValueError):
                        pass
            # An application listing alone is not enough; require an original AP submission.
            approved = any(
                s.get("submission_status") == "AP" and s.get("submission_type") == "ORIG"
                for s in row.get("submissions", [])
            )
            if not approved:
                counts["fda_without_original_approval"] += 1
                continue
            for product in row.get("products", []):
                ingredients = product.get("active_ingredients", [])
                # Combination approval cannot be transferred to each ingredient as standalone
                # approval.
                if len(ingredients) != 1:
                    counts["fda_combination_products_skipped"] += 1
                    continue
                name = ingredients[0].get("name", "").strip()
                if not name:
                    continue
                existing = registry.resolve(name, "drug")
                aliases = [product["brand_name"]] if product.get("brand_name") else []
                registry.add(
                    Entity(
                        id=existing.id if existing else drug_id(name),
                        kind="drug",
                        name=name,
                        aliases=aliases,
                        approved=True,
                        approved_since=min(dates) if dates else None,
                        identifiers={"fda_application": [row["application_number"]]},
                        sources=["openfda:drugsfda"],
                    )
                )

    # Explicit aliases let users reconcile salts, brands, and external source IDs without guessing.
    for path in files("local", ("aliases.csv",)):
        for row in table_rows(path):
            if row.get("entity_id") not in registry.entities or not row.get("alias"):
                raise ValueError(
                    f"{path}: aliases require an existing entity_id and a nonblank alias"
                )
            entity = registry.entities[row["entity_id"]].model_copy(deep=True)
            entity.aliases.append(row["alias"])
            entity.sources.append("local:aliases")
            registry.add(entity)

    # Stage 4: enrich only drugs already in the catalogue; do not add every research chemical.
    for path in files("b3db", (".tsv", ".tsv.gz")):
        for row in table_rows(path, "\t"):
            if not {"compound_name", "BBB+/BBB-"} <= row.keys():
                raise ValueError(f"{path}: expected B3DB compound_name and BBB+/BBB- columns")
            counts["b3db_rows"] += 1
            # Exact aliases control this join; unresolved names stay unmatched and are counted.
            entity = registry.resolve(row["compound_name"], "drug")
            label = {"BBB+": "positive", "BBB-": "negative"}.get(row["BBB+/BBB-"])
            if entity and label:
                enriched = entity.model_copy(deep=True)
                enriched.bbb = label
                enriched.sources = ["b3db"]
                if row.get("CID"):
                    enriched.identifiers["pubchem"] = [row["CID"]]
                registry.add(enriched)
                counts["b3db_matched_rows"] += 1

    # Stage 5: create independent pair judgments after catalogue/alias enrichment.
    judgments = []
    for row in repodb:
        drug = registry.resolve(row["drug_name"], "drug")
        disease = registry.resolve(row["ind_name"], "disease")
        counts["repodb_rows"] += 1
        status = row["status"].strip().lower()
        if not drug or not disease:
            counts["repodb_out_of_scope_or_unmatched"] += 1
            continue
        # This status mapping defines the benchmark, not a scientific claim of pharmacological
        # failure.
        relevance = {"approved": 1, "terminated": 0, "suspended": 0, "withdrawn": 0}.get(status)
        if relevance is not None:
            if row.get("ind_id"):
                enriched = disease.model_copy(deep=True)
                enriched.identifiers["umls"] = [row["ind_id"]]
                registry.add(enriched)
            judgments.append(
                Judgment(
                    drug_id=drug.id,
                    disease_id=disease.id,
                    relevance=relevance,
                    source="repodb",
                    status=status,
                )
            )
    for path in files("local", ("judgments.jsonl",)):
        for row in read_jsonl(path):
            judgment = Judgment.model_validate(row)
            if (
                judgment.drug_id not in registry.entities
                or registry.entities[judgment.drug_id].kind != "drug"
                or judgment.disease_id not in registry.entities
                or registry.entities[judgment.disease_id].kind != "disease"
            ):
                raise ValueError(f"{path}: judgment references unknown or wrongly typed entities")
            judgments.append(judgment)

    # Conflicting outcomes remain in the provenance file and are excluded from scoring later.
    unique_judgments = {j.model_dump_json(): j for j in judgments}
    judgments = list(unique_judgments.values())
    # Stage 6: validate, deduplicate, and link literature using the completed catalogue.
    documents = []
    seen_id, seen_doi, seen_text = set(), set(), set()
    document_paths = files("pubmed", (".xml", ".xml.gz", ".jsonl", ".jsonl.gz"))
    # Old managed batches are excluded; manually supplied files without manifests remain usable.
    active_path = raw / "pubmed" / "active-files.json"
    if active_path.exists():
        active = set(json.loads(active_path.read_text()))
        document_paths = [
            p
            for p in document_paths
            if not p.with_name(p.name + ".manifest.json").exists() or p.name in active
        ]
    document_paths += files("local", ("documents.jsonl", "documents.jsonl.gz"))
    for path in document_paths:
        iterator = (
            read_pubmed(path)
            if ".xml" in path.name
            else (Document.model_validate(row) for row in read_jsonl(path))
        )
        for doc in iterator:
            counts["documents_read"] += 1
            content_hash = text_hash(doc.text)
            doi = doc.doi.lower().strip() if doc.doi else None
            if doc.id in seen_id or (doi and doi in seen_doi) or content_hash in seen_text:
                counts["duplicate_documents"] += 1
                continue
            seen_id.add(doc.id)
            if doi:
                seen_doi.add(doi)
            seen_text.add(content_hash)
            # Explicit local links must reference real canonical entities instead of dangling IDs.
            unknown = set(doc.entity_ids) - registry.entities.keys()
            if unknown:
                raise ValueError(f"{path}: document {doc.id} has unknown entity IDs: {unknown}")
            # Text, MeSH names, and chemical names all contribute exact dictionary mentions.
            mentions = registry.mentions(" ".join([doc.text, *doc.mesh, *doc.chemicals]))
            doc.entity_ids = sorted(set(doc.entity_ids) | mentions)
            for name, mesh_id in doc.mesh_ids.items():
                disease = registry.resolve(name, "disease")
                if disease and mesh_id not in disease.identifiers.get("mesh", []):
                    enriched = disease.model_copy(deep=True)
                    enriched.identifiers["mesh"] = [mesh_id]
                    registry.add(enriched)
            counts["documents_without_date"] += doc.publication_date is None
            counts["documents_with_drug"] += any(
                registry.entities[i].kind == "drug" for i in doc.entity_ids
            )
            counts["documents_with_disease"] += any(
                registry.entities[i].kind == "disease" for i in doc.entity_ids
            )
            documents.append(doc)
    # Fail before writing processed outputs when no usable corpus was assembled.
    if not documents:
        raise ValueError("No documents found. Run drugir fetch or add documents under data/raw/.")
    # Stable output order keeps fingerprints and dense row alignment reproducible.
    documents.sort(key=lambda d: d.id)
    entities = sorted(registry.entities.values(), key=lambda e: e.id)
    judgments.sort(key=lambda j: (j.disease_id, j.drug_id, j.source, j.status))
    # Persist stage contracts separately; labels never become document text.
    write_jsonl(output / "entities.jsonl", entities)
    write_jsonl(output / "documents.jsonl", documents)
    write_jsonl(output / "judgments.jsonl", judgments)
    # Record coverage limitations so poor retrieval can be separated from missing inputs/joins.
    report = {
        "schema_version": 1,
        "counts": dict(counts),
        "documents": len(documents),
        "entities": len(entities),
        "drugs": sum(e.kind == "drug" for e in entities),
        "approved_drugs": sum(e.kind == "drug" and e.approved is True for e in entities),
        "drugs_with_bbb": sum(e.kind == "drug" and e.bbb is not None for e in entities),
        "judgments": len(judgments),
        "ambiguous_aliases": registry.ambiguous_aliases,
        "disease_document_counts": {
            d.id: sum(d.id in doc.entity_ids for doc in documents) for d in settings.diseases
        },
        "inputs": [{"path": str(p.relative_to(raw)), "sha256": digest(p)} for p in inputs],
        "limitations": [
            "Dictionary linking uses exact aliases; salt/brand ambiguity remains explicit.",
            "repoDB approved indications are recovery labels, not verified novel repurposings.",
            "B3DB evidence dates are unavailable; BBB reranking is retrospective in temporal runs.",
        ],
    }
    write_json(output / "ingestion-report.json", report)
    return report
