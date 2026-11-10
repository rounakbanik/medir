"""Validated records shared by ingestion, indexing, search, and evaluation.

Read this module first when joining the project. These Pydantic models define
the JSONL files on disk and the JSON returned by the CLI/API. A canonical
entity, a literature document, and a relevance judgment are different records:
keeping them separate prevents evaluation outcomes from becoming search text.
Optional metadata uses None for unknown information rather than guessing it.
"""

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Record(BaseModel):
    """Common base for strict, serializable pipeline records.

    Unknown fields are rejected so a misspelled key or changed upstream schema
    fails visibly. Mutable defaults in subclasses use Field(default_factory=...)
    so separate records never accidentally share the same list or dictionary.
    """

    model_config = ConfigDict(extra="forbid")


class Entity(Record):
    """One canonical drug or disease, with aliases and source metadata.

    Attributes:
        id: Stable local identifier used to join every pipeline stage.
        kind: Whether this entity is a drug or a disease.
        name: Preferred human-readable label; aliases supply alternative names.
        identifiers: External namespace to identifier-list mappings, such as MeSH.
        cns: Whether a disease should activate the CNS feasibility rule.
        approved: Known approval signal, or None when approval is unknown.
        approved_since: Earliest observed original approval date, if available.
        bbb: Matched BBB evidence; conflicting measurements remain explicit.
        sources: Provenance labels describing where metadata originated.

    Approval and BBB information describe the entity; neither establishes that
    it treats a particular disease. Salt forms remain distinct unless mapped.
    """

    # This local key, rather than a display name, is used for joins across files.
    id: str
    kind: Literal["drug", "disease"]
    name: str = Field(min_length=1)
    aliases: list[str] = Field(default_factory=list)
    # A namespace can contain multiple observed IDs; provenance is preserved.
    identifiers: dict[str, list[str]] = Field(default_factory=dict)
    # Only disease records use this switch to activate the CNS/BBB pair rule.
    cns: bool = False
    # None means missing evidence; it is deliberately different from False.
    approved: bool | None = None
    approved_since: date | None = None
    # Keep contradictory measurements explicit instead of picking a convenient label.
    bbb: Literal["positive", "negative", "conflicting"] | None = None
    sources: list[str] = Field(default_factory=list)


class Document(Record):
    """One retrievable literature record, usually a PubMed abstract.

    MeSH and chemical names are retained together with their external IDs.
    entity_ids contains the canonical mentions supplied or found by linking.
    publication_date may be absent; date_precision records whether its day is
    exact or represents the conservative end of a known month/year interval.
    Local additions use this same schema as source-adapted PubMed documents.
    """

    # This local key, rather than a display name, is used for joins across files.
    id: str
    title: str = ""
    abstract: str = Field(min_length=1)
    # Date precision is stored separately so an imputed end-of-year is visible.
    publication_date: date | None = None
    date_precision: Literal["day", "month", "year", "unknown"] = "unknown"
    mesh: list[str] = Field(default_factory=list)
    mesh_ids: dict[str, str] = Field(default_factory=dict)
    chemicals: list[str] = Field(default_factory=list)
    chemical_ids: dict[str, str] = Field(default_factory=dict)
    # These refer to Entity.id values; they are mentions, not treatment assertions.
    entity_ids: list[str] = Field(default_factory=list)
    doi: str | None = None
    source: str = "local"
    url: str | None = None

    @property
    def text(self) -> str:
        """Return title plus abstract as the raw text used by neural models.

        A newline preserves the title/abstract boundary. MeSH terms are added
        separately to the lexical index; they are not part of this property.
        """
        return f"{self.title}\n{self.abstract}"


class Judgment(Record):
    """An independently sourced binary relevance label for a drug/disease pair.

    relevance=1 means relevant under the benchmark; relevance=0 means a judged
    negative. No record means unjudged, not a proven failure. Temporal fields
    must describe reviewed historical events and an earlier different indication.
    RepoDB status alone does not establish those dates or a novel repurposing.
    """

    drug_id: str
    disease_id: str
    # An absent judgment is represented by no record, not a fabricated zero label.
    relevance: Literal[0, 1]
    source: str
    status: str = ""
    confirmed_date: date | None = None
    prior_indication_date: date | None = None
    # A manually verified prior DIFFERENT indication, not merely a drug approval date.
    prior_indication: str | None = None


class Evidence(Record):
    """A bounded, inspectable paper reference attached to one search result.

    score is the paper retrieval/reranking score, not the final entity score.
    snippet is an extracted sentence, possibly truncated, rather than generated
    scientific text. A referenced paper can mention a pair without endorsing it.
    """

    document_id: str
    title: str
    publication_date: date | None
    url: str | None
    score: float
    snippet: str


class SearchResult(Record):
    """A ranked entity with its score components and supporting literature.

    retrieval_score is normalized within the current query. score adds the
    feasibility_adjustment, so it can exceed one and is not a probability.
    evidence_count counts matched papers in the retrieved candidate pool; the
    evidence list stores only the configured number of top paper references.
    The random baseline can return entities with zero evidence by design.
    """

    entity: Entity
    score: float
    # Store the unadjusted entity score so the BBB effect can be inspected.
    retrieval_score: float
    feasibility_adjustment: float = 0
    # This can exceed len(evidence), because only a few paper excerpts are displayed.
    evidence_count: int
    evidence: list[Evidence]


class SearchResponse(Record):
    """Complete serializable answer to a single search request.

    resolved_entity is None for an unresolved mechanism query. Warnings retain
    qualifications such as missing indexed mentions or retrospective BBB labels.
    """

    query: str
    direction: Literal["disease-to-drug", "drug-to-disease", "mechanism"]
    method: str
    resolved_entity: str | None
    results: list[SearchResult]
    warnings: list[str] = Field(default_factory=list)


class SearchRequest(Record):
    """Validated search parameters shared by the engine and HTTP endpoint.

    Entity directions require a resolvable name or canonical ID; mechanism
    direction accepts free text. k limits the number of returned entities,
    independently of the paper-candidate depth configured for retrieval.
    """

    query: str = Field(min_length=1, max_length=2000)
    direction: Literal["disease-to-drug", "drug-to-disease", "mechanism"] = "disease-to-drug"
    method: Literal[
        "random",
        "bm25",
        "dense",
        "hybrid",
        "cross-encoder",
        "abc",
        "tf-idf",
        "dirichlet",
        "pivoted",
        "word-count",
        "custom",
    ] = "bm25"
    # The paper pool is configured elsewhere; k limits the final entity list.
    k: int = Field(default=10, ge=1, le=100)
    feasibility: bool = False

    @model_validator(mode="after")
    def nonblank(self):
        """Reject whitespace-only queries that satisfy a simple length check.

        Returns the validated instance so Pydantic can complete its after-validator.
        Actual whitespace trimming happens in SearchEngine.search.
        """
        if not self.query.strip():
            raise ValueError("Query must contain non-whitespace text")
        return self
