"""Create reusable literature snapshots and optional dense embedding matrices.

The lexical snapshot stores documents, entities, and token lists as JSONL;
SearchEngine reconstructs the rank-bm25 object when loaded. Dense embeddings
are float32 NumPy rows in exactly the snapshot document order. Manifests
record source/settings fingerprints so incompatible artifacts are rejected.
"""

import hashlib
import json
import os
import re
from datetime import date
from pathlib import Path

import numpy as np

from .config import Settings
from .io import digest, read_jsonl, write_json, write_jsonl
from .models import Document, Entity


def tokenize(text: str) -> list[str]:
    """Lowercase text and extract lexical tokens while retaining internal hyphens/apostrophes.

    This BM25 tokenizer is separate from entity-name normalization and neural
    tokenization. Neural models receive raw title/abstract strings instead.
    """
    return re.findall(r"\b\w+(?:[-']\w+)*\b", text.lower())


def signature(value) -> str:
    """Hash a deterministically serialized settings/fingerprint structure.

    Sorting dictionary keys makes equivalent mappings independent of insertion
    order. This function hashes metadata; io.digest hashes file bytes.
    """
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def input_signature(settings: Settings):
    """Collect the normalized data hashes and settings that define the lexical snapshot.

    Source documents/entities, the literature cutoff, and BM25 parameters change
    snapshot compatibility. Judgment labels do not: they are evaluated separately
    and are not needed to build a searchable literature index.
    """
    # Hash the prepared contracts rather than every raw source on each search startup.
    return {
        "schema_version": 1,
        "documents": digest(settings.paths.processed / "documents.jsonl"),
        "entities": digest(settings.paths.processed / "entities.jsonl"),
        "cutoff_year": settings.evaluation.cutoff_year,
        "bm25_k1": settings.retrieval.bm25_k1,
        "bm25_b": settings.retrieval.bm25_b,
    }


def neural_environment(settings: Settings):
    """Set project-local model-cache defaults without overriding user environment choices.

    Call before importing neural libraries because they may read cache settings
    during import. setdefault preserves a teammate's existing HF_HOME setting.
    """
    # Respect user-supplied caches, otherwise keep large downloads inside the project.
    os.environ.setdefault("HF_HOME", str(settings.paths.index.parent / "models"))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def encoder(settings: Settings, revision_override=None):
    """Load the configured SentenceTransformer only when neural retrieval is requested.

    Args:
        settings: Model name/revision, target device, and sequence-length settings.
        revision_override: Snapshot revision recorded with existing embeddings;
            used only if configuration does not explicitly pin a revision.

    Returns:
        A SentenceTransformer ready to encode raw title/abstract strings.

    Raises:
        RuntimeError: Optional neural dependencies have not been installed.

    First use may download model weights. Device compatibility and model-specific
    maximum sequence limits remain the caller's configuration responsibility.
    """
    # Keep neural imports lazy so base/BM25 installations do not need PyTorch.
    neural_environment(settings)
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise RuntimeError("Install neural models with: pip install -e '.[neural]'") from error
    r = settings.retrieval
    model = SentenceTransformer(
        r.dense_model, revision=r.dense_revision or revision_override, device=r.device
    )
    # Long documents are truncated by the model tokenizer at this configured limit.
    model.max_seq_length = r.max_sequence_length
    return model


def dense_signature(settings, index_hash):
    """Describe which corpus, model, revision, and sequence limit dense rows belong to.

    Query-only controls such as displayed k do not invalidate document embeddings.
    The index hash includes entity metadata too, so even an entity-only change
    currently invalidates dense compatibility in this conservative v1.
    """
    r = settings.retrieval
    return {
        "index_hash": index_hash,
        "model": r.dense_model,
        "revision": r.dense_revision,
        "max_sequence_length": r.max_sequence_length,
    }


def cached_model_revision(model_name, revision=None):
    """Find the actual Hugging Face snapshot commit used by a cached model.

    Newer Transformers versions do not always expose config._commit_hash, so
    inspect the cache's snapshots/<commit>/config.json path instead. For a local
    model directory there is no remote commit; return None. If cache resolution
    cannot provide a commit, return the explicitly supplied revision if any.
    """
    from huggingface_hub import try_to_load_from_cache

    if Path(model_name).is_dir():
        return None
    cached = try_to_load_from_cache(model_name, "config.json", revision=revision or "main")
    if isinstance(cached, str):
        parent = Path(cached).parent
        if parent.parent.name == "snapshots":
            return parent.name
    return revision


def build_index(settings: Settings, dense=False):
    """Build or reuse the lexical snapshot and optionally its dense embeddings.

    Args:
        settings: Paths, date restrictions, BM25 controls, and neural settings.
        dense: Also build/validate the document embedding matrix when true.

    Returns:
        The persisted lexical manifest, including index_hash and document count.

    Side effects:
        Writes snapshot JSONL files and a manifest; dense mode may download a
        model and writes embeddings.npy plus dense.json. Compatible artifacts
        are reused. Incompatible dense artifacts are rebuilt as a full matrix.

    Raises:
        ValueError: No documents remain after optional date filtering.

    Unknown publication dates are excluded when a cutoff is active. Dense row
    order must stay aligned with documents.jsonl; do not reorder one separately.
    """
    target = settings.paths.index
    target.mkdir(parents=True, exist_ok=True)
    inputs = input_signature(settings)
    index_hash = signature(inputs)
    manifest_path = target / "manifest.json"
    # A matching manifest allows the existing snapshot to be reused.
    old = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    # Rebuild the full lexical snapshot when its defining inputs/settings change.
    if old.get("index_hash") != index_hash:
        docs = [
            Document.model_validate(r)
            for r in read_jsonl(settings.paths.processed / "documents.jsonl")
        ]
        original_count = len(docs)
        # Date filtering happens before tokenization and dense encoding, not just at display time.
        if settings.evaluation.cutoff_year:
            cutoff = date(settings.evaluation.cutoff_year, 12, 31)
            docs = [d for d in docs if d.publication_date and d.publication_date <= cutoff]
        if not docs:
            raise ValueError("No dated documents remain under the configured cutoff")
        entities = [
            Entity.model_validate(r)
            for r in read_jsonl(settings.paths.processed / "entities.jsonl")
        ]
        # This order is the canonical embedding-row order; never sort a matrix independently.
        write_jsonl(target / "documents.jsonl", docs)
        write_jsonl(target / "entities.jsonl", entities)
        write_jsonl(
            target / "tokens.jsonl", (tokenize(d.text + " " + " ".join(d.mesh)) for d in docs)
        )
        write_json(
            manifest_path,
            {
                "index_hash": index_hash,
                "inputs": inputs,
                "documents": len(docs),
                "excluded_by_date": original_count - len(docs),
            },
        )
    # Dense mode is optional; lexical snapshots are sufficient for BM25/ABC/random methods.
    if dense:
        expected = dense_signature(settings, index_hash)
        dense_meta = target / "dense.json"
        cached = json.loads(dense_meta.read_text()) if dense_meta.exists() else {}
        if cached.get("signature") != expected or not (target / "embeddings.npy").exists():
            docs = [Document.model_validate(r) for r in read_jsonl(target / "documents.jsonl")]
            model = encoder(settings)
            # Normalize rows so query/document dot products are cosine similarities.
            embeddings = model.encode(
                [d.text for d in docs],
                batch_size=settings.retrieval.batch_size,
                normalize_embeddings=True,
                show_progress_bar=True,
            )
            temp = target / "embeddings.tmp.npy"
            # Float32 reduces storage; allow_pickle=False keeps the saved format purely numeric.
            np.save(temp, np.asarray(embeddings, dtype=np.float32), allow_pickle=False)
            temp.replace(target / "embeddings.npy")
            commit = cached_model_revision(
                settings.retrieval.dense_model, settings.retrieval.dense_revision
            )
            write_json(
                dense_meta,
                {
                    "signature": expected,
                    "shape": list(embeddings.shape),
                    "resolved_revision": commit,
                    "sha256": digest(target / "embeddings.npy"),
                },
            )
        # An older compatible matrix may only need its cached model commit recorded.
        elif not cached.get("resolved_revision"):
            neural_environment(settings)
            cached["resolved_revision"] = cached_model_revision(
                settings.retrieval.dense_model, settings.retrieval.dense_revision
            )
            write_json(dense_meta, cached)
    return json.loads(manifest_path.read_text())
