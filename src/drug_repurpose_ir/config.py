"""Load and validate the project's user-editable YAML configuration.

Settings are grouped by pipeline responsibility rather than hidden inside
model classes. The CLI calls load_config; tests may construct Settings
directly with temporary absolute paths. Keep experiment files immediately
inside configs/: relative data paths resolve from the config grandparent.
"""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field

from .models import Entity, Record


class Paths(Record):
    """Locations of raw inputs, normalized records, index snapshots, and runs.

    load_config resolves these paths to absolute locations. The raw/processed
    separation lets teammates inspect source files without editing derived data.
    """

    raw: Path = Path("data/raw")
    processed: Path = Path("data/processed")
    index: Path = Path("artifacts/index")
    runs: Path = Path("runs")


class Acquisition(Record):
    """Download scope, PubMed sampling controls, and public dataset locations.

    PubMed limits apply separately to each disease and extra query. The B3DB
    URL pins a source commit, while the Figshare article identifies the archived
    repoDB release. This is a bounded starter sample, not an exhaustive corpus.
    """

    # Cap each search job; extra anchor queries receive their own cap too.
    pubmed_per_disease: int = Field(default=100, ge=1, le=9999)
    pubmed_batch_size: int = Field(default=100, ge=1, le=500)
    pubmed_extra_query: str = '("drug therapy"[Subheading] OR repurpos*[Title/Abstract])'
    pubmed_sort: Literal["relevance", "pub date"] = "relevance"
    pubmed_start_year: int = Field(default=1990, ge=1800, le=2100)
    pubmed_end_year: int = Field(default=2026, ge=1800, le=2100)
    # Named extra searches support anchor cases without changing acquisition code.
    pubmed_queries: dict[str, str] = Field(default_factory=dict)
    # Pin this dataset to a source commit for reproducible raw inputs.
    b3db_url: str = (
        "https://raw.githubusercontent.com/theochem/B3DB/"
        "75dab1cc607bb7a03f3de5c576cffd223e767844/B3DB/B3DB_classification.tsv"
    )
    repodb_article_id: int = 3811674


class Retrieval(Record):
    """Model, scoring, eligibility, and evidence-display choices for search.

    Paper-candidate depth controls the first-stage pool, not final entity k.
    Dense and cross-encoder revisions may be pinned for experiments. CPU is the
    portable default; mps/cuda require compatible local hardware and PyTorch.
    BBB weight is a heuristic score adjustment, not a learned clinical effect.
    """

    # This is first-stage paper depth, not the number of final drug/disease results.
    candidate_documents: int = Field(default=200, ge=1)
    evidence_per_entity: int = Field(default=3, ge=1)
    # Choose how several papers contribute to one candidate entity.
    aggregation: Literal["max", "sum", "top-mean"] = "max"
    aggregation_top_n: int = Field(default=3, ge=1)
    # These defaults match the supplied BM25 implementation.
    bm25_k1: float = Field(default=1.2, gt=0)
    bm25_b: float = Field(default=0.75, ge=0, le=1)
    bm25_k3: float = Field(default=8, ge=0)
    index_type: Literal["basic", "positional"] = "basic"
    token_regex: str = r"\w+"
    stopwords: list[str] = Field(default_factory=list)
    minimum_word_frequency: int = Field(default=0, ge=0)
    dirichlet_mu: float = Field(default=2000, gt=0)
    pivoted_b: float = Field(default=0.2, ge=0, le=1)
    custom_alpha: float = Field(default=0.5, ge=0)
    # Larger RRF offsets reduce the influence of very high first-stage ranks.
    rrf_k: int = Field(default=60, ge=1)
    dense_model: str = "NeuML/pubmedbert-base-embeddings"
    # Optional model commit pin; document embeddings also record the resolved commit.
    dense_revision: str | None = None
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    cross_encoder_revision: str | None = None
    device: Literal["cpu", "mps", "cuda"] = "cpu"
    batch_size: int = Field(default=16, ge=1)
    max_sequence_length: int = Field(default=512, ge=32, le=8192)
    # Unknown approval evidence excludes a drug when this switch is enabled.
    approved_only: bool = True
    # This is a manually chosen score adjustment, not a probability or learned weight.
    bbb_weight: float = Field(default=0.15, ge=0, le=1)
    seed: int = 650


class Evaluation(Record):
    """Metric cutoffs, uncertainty settings, and optional temporal restrictions.

    A literature cutoff also affects indexing even in indication-recovery mode.
    Temporal-repurposing mode additionally requires reviewed dated pair labels.
    Set bootstrap_samples to zero when only point estimates are required.
    """

    ks: list[int] = Field(default_factory=lambda: [5, 10, 20])
    bootstrap_samples: int = Field(default=1000, ge=0, le=100000)
    # None uses all prepared papers; a year activates historical literature filtering.
    cutoff_year: int | None = None
    mode: Literal["indication-recovery", "temporal-repurposing"] = "indication-recovery"


class Settings(Record):
    """Root validated configuration passed explicitly between pipeline stages.

    Each nested default is constructed independently. The disease catalogue
    defines candidate scope and exact aliases; it is not inferred from labels.
    """

    schema_version: int = 1
    paths: Paths = Field(default_factory=Paths)
    acquisition: Acquisition = Field(default_factory=Acquisition)
    retrieval: Retrieval = Field(default_factory=Retrieval)
    evaluation: Evaluation = Field(default_factory=Evaluation)
    diseases: list[Entity] = Field(default_factory=list)


def load_config(path: Path | str = "configs/default.yaml") -> Settings:
    """Read YAML, validate settings, and anchor configured filesystem paths.

    Args:
        path: YAML file, normally configs/default.yaml or configs/<experiment>.yaml.

    Returns:
        Settings with absolute data/index/run paths and validated study controls.

    Raises:
        ValueError: Cutoffs, disease IDs/types, or year ranges are inconsistent.
        OSError: The requested YAML file cannot be read.
        pydantic.ValidationError: A field has an invalid type/value or is unknown.

    The project root is path.parent.parent. Moving a config into a deeper
    directory therefore changes relative-path interpretation; use absolute paths
    or keep experiment configurations directly under configs/.
    """
    path = Path(path).resolve()
    # Schema validation catches misspelled keys and invalid parameter types early.
    settings = Settings.model_validate(yaml.safe_load(path.read_text()) or {})
    # Paths are anchored at the project config's parent directory, not the shell cwd.
    root = path.parent.parent
    # Resolve every path consistently so commands share the same data locations.
    for key in type(settings.paths).model_fields:
        value = getattr(settings.paths, key)
        setattr(settings.paths, key, (root / value).resolve())
    # Cross-field checks below complement Pydantic's individual-field validation.
    if not settings.evaluation.ks or any(k < 1 or k > 100 for k in settings.evaluation.ks):
        raise ValueError("evaluation.ks must contain integers between 1 and 100")
    if len({d.id for d in settings.diseases}) != len(settings.diseases):
        raise ValueError("Duplicate disease IDs in configuration")
    if any(d.kind != "disease" for d in settings.diseases):
        raise ValueError("All configured diseases must have kind=disease")
    a = settings.acquisition
    if a.pubmed_start_year > a.pubmed_end_year:
        raise ValueError("PubMed start year must not exceed end year")
    if settings.evaluation.mode == "temporal-repurposing" and not settings.evaluation.cutoff_year:
        raise ValueError("Temporal evaluation requires evaluation.cutoff_year")
    return settings
