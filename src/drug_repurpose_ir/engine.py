"""Turn a query into ranked entities with inspectable literature evidence.

Direct methods rank papers first, require query/candidate co-mentions, and
aggregate those paper scores into entities. Random and MeSH-bridge ABC are
comparison baselines. The optional BBB rule adjusts pair feasibility after
aggregation; it does not classify therapeutic versus adverse/background
mentions. Search scores must not be interpreted as efficacy probabilities.
"""

import hashlib
import json
import re
from collections import Counter, defaultdict

import numpy as np

from .config import Settings
from .entities import Registry, normalize
from .indexing import (
    cached_model_revision,
    dense_signature,
    input_signature,
    neural_environment,
    signature,
    tokenize,
)
from .io import read_jsonl
from .models import Document, Entity, Evidence, SearchRequest, SearchResponse, SearchResult
from .reference_adapter import (
    LEXICAL_METHODS,
    lexical_query,
    load_reference_index,
    reference_cross_encoder,
    reference_ranker,
    reference_vector_ranker,
)


def rrf(rankings: list[list[int]], constant: int = 60) -> dict[int, float]:
    """Fuse ordered document lists using reciprocal rank fusion (RRF).

    Args:
        rankings: Lists of document row indices, most relevant first.
        constant: Positive rank offset; each occurrence contributes 1/(offset+rank).

    Returns:
        Document row index to summed fusion score. Ranks are one-based.

    Fusion uses ordering rather than raw BM25/dot-product values, whose numerical
    scales differ. Sorting the returned mapping is the caller's responsibility.
    """
    # A document can receive contributions from both lexical and dense rank lists.
    scores = defaultdict(float)
    for ranking in rankings:
        for rank, item in enumerate(ranking, 1):
            scores[item] += 1 / (constant + rank)
    return dict(scores)


def aggregate(scores, mode, top_n):
    """Reduce one entity's nonempty supporting-paper scores to a single score.

    max uses the strongest paper; sum rewards all matched papers; top-mean
    averages up to top_n highest scores. The configuration validates mode and
    top_n. This helper assumes at least one supporting score is present.
    """
    ordered = sorted(scores, reverse=True)
    if mode == "max":
        return ordered[0]
    if mode == "sum":
        return sum(ordered)
    return float(np.mean(ordered[:top_n]))


class SearchEngine:
    """Load one immutable index snapshot and reuse its search resources.

    The engine keeps BM25, catalogue/linking state, and paper/entity/MeSH maps in
    memory. The supplied VectorRanker and the embedding matrix load lazily.
    Cached dense scores and a bounded paper-ranking cache reduce repeated evaluation
    work. Create a new engine after changing data/settings; it is not a watcher.
    """

    def __init__(self, settings: Settings):
        """Validate snapshot compatibility and initialize lexical/search lookup state.

        Args:
            settings: Validated configuration matching the prepared data/index.

        Raises:
            ValueError: Index files are missing or the data/settings fingerprint changed.

        Neural weights are not loaded here. The supplied modules import their
        neural libraries, but BM25 does not instantiate or download a model.
        """
        self.settings = settings
        # Set cache defaults before the reference modules import neural libraries.
        neural_environment(settings)
        target = settings.paths.index
        if not (target / "manifest.json").exists():
            raise ValueError("Index missing. Run drugir prepare then drugir index.")
        self.manifest = json.loads((target / "manifest.json").read_text())
        if signature(input_signature(settings)) != self.manifest["index_hash"]:
            raise ValueError("Index is stale. Run drugir index after changing data or cutoff.")
        self.documents = [
            Document.model_validate(r) for r in read_jsonl(target / "documents.jsonl")
        ]
        self.registry = Registry(
            Entity.model_validate(r) for r in read_jsonl(target / "entities.jsonl")
        )
        # Restore the supplied inverted index and use its Ranker/BM25 classes directly.
        self.document_index = load_reference_index(settings)
        if set(self.document_index.document_metadata) != set(range(len(self.documents))):
            raise ValueError("Lexical docid/document alignment is invalid; rebuild index")
        self.ranker = reference_ranker(self.document_index, settings)
        self._lexical_rankers = {"bm25": self.ranker}
        self._vector_ranker = None
        self._cross_encoder = None
        self._embeddings = None
        self._dense_revision = None
        self.model_revisions = {}
        # Cache reference VectorRanker outputs across dense/hybrid/BBB variants.
        self._dense_scores = {}
        self._retrieval_cache = {}
        # Reverse lookup maps support coverage reporting and the indirect MeSH-bridge baseline.
        self._entity_docs = defaultdict(set)
        self._mesh_docs = defaultdict(set)
        for i, doc in enumerate(self.documents):
            for eid in doc.entity_ids:
                self._entity_docs[eid].add(i)
            for term in doc.mesh:
                self._mesh_docs[normalize(term)].add(i)

    def candidate_allowed(self, entity: Entity) -> bool:
        """Check approval and optional historical availability for a result entity.

        Drug approval must be explicitly true when approved_only is enabled. With
        a cutoff, a drug also needs an observed approval date no later than that
        year. Diseases are not subject to drug-approval filtering. This checks result
        eligibility; entity-direction query resolution is a separate operation.
        """
        if (
            entity.kind == "drug"
            and self.settings.retrieval.approved_only
            and entity.approved is not True
        ):
            return False
        # Even an approved drug needs a known historical approval date in cutoff runs.
        cutoff = self.settings.evaluation.cutoff_year
        if entity.kind == "drug" and cutoff:
            return entity.approved_since is not None and entity.approved_since.year <= cutoff
        return True

    def candidate_ids(self, kind: str):
        """Return the full eligible catalogue IDs for one requested entity kind.

        Entities without indexed papers remain in this universe. Random retrieval
        can rank them, and evaluation can expose coverage failures instead of hiding
        difficult positives from its denominator.
        """
        # Include approved catalog entities without documents: random is a true entity baseline,
        # and evaluation does not hide corpus/normalization failures from the Recall denominator.
        return {
            e.id
            for e in self.registry.entities.values()
            if e.kind == kind and self.candidate_allowed(e)
        }

    def _dense(self, text):
        """Get scores from the supplied VectorRanker against aligned document embeddings.

        Lazily validate/load embeddings.npy and reload the model revision recorded
        with those rows. VectorRanker.query retains its original dot product and
        query encoding. Cache each score vector for reuse in this engine.

        Returns:
            One score per document row, in documents.jsonl order.

        Raises:
            ValueError: Dense metadata is missing, stale, or row alignment is invalid.
        """
        target = self.settings.paths.index
        if self._embeddings is None:
            meta_path = target / "dense.json"
            if not meta_path.exists():
                raise ValueError("Dense index missing. Run drugir index --dense.")
            meta = json.loads(meta_path.read_text())
            expected = dense_signature(self.settings, self.manifest["index_hash"])
            if meta["signature"] != expected:
                raise ValueError("Dense embeddings are stale. Run drugir index --dense.")
            self._dense_revision = meta.get("resolved_revision")
            self.model_revisions["dense"] = self._dense_revision
            # Memory-map numeric rows so loading the matrix need not duplicate it in RAM.
            self._embeddings = np.load(target / "embeddings.npy", mmap_mode="r", allow_pickle=False)
            if self._embeddings.shape[0] != len(self.documents):
                raise ValueError("Dense row/document alignment is invalid; rebuild index")
        # Use the document matrix's resolved snapshot to prevent model/version mismatch.
        if self._vector_ranker is None:
            self._vector_ranker = reference_vector_ranker(
                self.settings, self._embeddings, self._dense_revision
            )
        if text not in self._dense_scores:
            # query() performs the supplied encoding, dot product, and sorting.
            ranking = self._vector_ranker.query(text)
            scores = np.zeros(len(self.documents), dtype=float)
            for docid, score in ranking:
                scores[docid] = score
            self._dense_scores[text] = scores
        return self._dense_scores[text]

    def _top(self, scores, limit, positive_only=False):
        """Select deterministic top document rows from a score vector.

        Args:
            scores: Vector aligned with self.documents.
            limit: Maximum number of document row indices to return.
            positive_only: Exclude zero/negative lexical matches when true.

        Ties use document ID rather than nondeterministic ordering. A full sort is
        intentionally simple for the starter corpus; no approximate index is used.
        """
        indices = np.arange(len(scores))
        if positive_only:
            indices = indices[np.asarray(scores) > 0]
        # Deterministic ties by document order. Full sort is appropriate for a local v1 corpus.
        return sorted(indices.tolist(), key=lambda i: (-float(scores[i]), self.documents[i].id))[
            :limit
        ]

    def retrieve(self, text, method):
        """Return cached paper rankings, computing them only when absent.

        The per-instance cache holds at most 256 (query text, method) entries and
        refreshes their recency when reused. Callers treat returned ranking lists as
        read-only. Feasibility changes entity scores later and can reuse these papers.
        """
        # Cache by exact expanded query and method; a BBB toggle can reuse first-stage papers.
        key = (text, method)
        if key in self._retrieval_cache:
            value = self._retrieval_cache.pop(key)
            self._retrieval_cache[key] = value
            return value
        value = self._retrieve(text, method)
        if len(self._retrieval_cache) >= 256:
            self._retrieval_cache.pop(next(iter(self._retrieval_cache)))
        self._retrieval_cache[key] = value
        return value

    def _retrieve(self, text, method):
        """Run BM25, dense, hybrid, or cross-encoder paper retrieval.

        BM25 and dense produce method-specific rankings. Hybrid applies RRF to each
        first-stage list and retains candidate_documents papers. Cross-encoder uses
        that hybrid pool through the supplied CrossEncoderScorer.score method.
        Its raw paper scores are retained as evidence, not treatment probabilities.

        Returns:
            Ordered (document_row_index, score) pairs. Query-entity co-mention checks
            happen later in search, after this paper-candidate pool is chosen.
        """
        r = self.settings.retrieval
        if method in LEXICAL_METHODS:
            if method not in self._lexical_rankers:
                self._lexical_rankers[method] = reference_ranker(
                    self.document_index, self.settings, method
                )
            # Negative BM25 scores are valid in the reference formula; do not discard them.
            return lexical_query(self._lexical_rankers[method], text, method)[
                : r.candidate_documents
            ]
        lexical = self.retrieve(text, "bm25")
        dense = self._dense(text)
        if method == "dense":
            return [(i, float(dense[i])) for i in self._top(dense, r.candidate_documents)]
        lexical_ids = [docid for docid, _ in lexical]
        dense_ids = self._top(dense, r.candidate_documents)
        # Raw lexical and semantic scores are not directly added; their ranks are fused.
        combined = rrf([lexical_ids, dense_ids], r.rrf_k)
        candidates = sorted(combined, key=lambda i: (-combined[i], self.documents[i].id))
        candidates = candidates[: r.candidate_documents]
        # The expensive joint model only scores the bounded hybrid candidate pool.
        if method == "cross-encoder":
            if self._cross_encoder is None:
                neural_environment(self.settings)
                self._cross_encoder = reference_cross_encoder(self.settings, self.documents)
                self.model_revisions["cross_encoder"] = cached_model_revision(
                    r.cross_encoder_model, r.cross_encoder_revision
                )
            values = [self._cross_encoder.score(i, text) for i in candidates]
            return sorted(
                zip(candidates, map(float, values), strict=True),
                key=lambda pair: (-pair[1], pair[0]),
            )
        return [(i, combined[i]) for i in candidates]

    def _abc(self, entity, target_kinds):
        """Score indirect A-to-B-to-C associations using shared non-entity MeSH terms.

        A is the resolved query entity, B is a shared MeSH term, and C is a candidate
        entity. Candidate evidence comes from bridge papers outside A's own paper
        set. Weighting combines query-side occurrence, bridge specificity, and
        candidate paper count. Shared terms are descriptive links, not causal proof.

        Returns:
            (entity_scores, supporting_document_scores) dictionaries for search.

        Raises:
            ValueError: No unambiguous query entity was resolved.
        """
        if entity is None:
            raise ValueError("ABC requires a resolvable drug or disease query")
        query_docs = self._entity_docs[entity.id]
        # B terms are shared MeSH descriptors that are not themselves resolved entity names.
        bridges = Counter()
        # MeSH co-occurrence is a transparent A->B->C baseline, not mechanistic inference.
        for i in query_docs:
            bridges.update(
                normalize(t) for t in set(self.documents[i].mesh) if not self.registry.resolve(t)
            )
        scores, supports = defaultdict(float), defaultdict(dict)
        for term, count in bridges.items():
            if not term:
                continue
            docs = self._mesh_docs[term]
            idf = np.log(1 + len(self.documents) / max(len(docs), 1))
            weight = count / max(len(query_docs), 1) * idf
            # Exclude direct query papers: this baseline measures indirect shared terms.
            for i in docs - query_docs:
                for eid in self.documents[i].entity_ids:
                    candidate = self.registry.entities[eid]
                    if candidate.kind in target_kinds and self.candidate_allowed(candidate):
                        scores[eid] += weight / max(len(self._entity_docs[eid]), 1)
                        supports[eid][i] = supports[eid].get(i, 0) + float(weight)
        return scores, supports

    def search(
        self, query: str, direction="disease-to-drug", method="bm25", k=10, feasibility=False
    ):
        """Resolve a query and return ranked drugs/diseases with supporting papers.

        Args:
            query: Entity name/alias/ID, or free text for mechanism direction.
            direction: disease-to-drug, drug-to-disease, or mechanism.
            method: random, bm25, dense, hybrid, cross-encoder, or abc.
            k: Maximum final entities to return, between 1 and 100.
            feasibility: Apply the configured BBB adjustment to CNS pairs when true.

        Returns:
            SearchResponse containing canonical entities, normalized score components,
            bounded evidence excerpts, and applicable coverage/metadata warnings.

        Direct methods require both entities in a selected paper. Aggregate scores
        are divided by the highest entity score for this query, then BBB adjustments
        are added. Unknown/conflicting BBB labels remain neutral. Mechanism queries
        without a disease context receive no pair-specific BBB adjustment.

        Raises:
            ValueError: An entity query is unknown/ambiguous or a required index is stale.
            pydantic.ValidationError: Request values such as k/method are invalid.
        """
        request = SearchRequest(
            query=query, direction=direction, method=method, k=k, feasibility=feasibility
        )
        query = request.query.strip()
        # Step 1: resolve typed entity queries; mechanism direction leaves free text available.
        kind = {"disease-to-drug": "disease", "drug-to-disease": "drug"}.get(direction)
        entity = self.registry.resolve(query, kind)
        if kind and entity is None:
            raise ValueError(f"Query is unknown or ambiguous as a {kind}; add an explicit alias")
        targets = (
            {"drug"}
            if direction == "disease-to-drug"
            else ({"disease"} if direction == "drug-to-disease" else {"drug", "disease"})
        )
        warnings = []
        # Step 2: baselines and direct retrieval have different ways to propose candidates.
        if method == "random":
            seed = int(
                hashlib.sha256(f"{self.settings.retrieval.seed}:{query}".encode()).hexdigest()[:16],
                16,
            )
            rng = np.random.default_rng(seed)
            ids = sorted(set().union(*(self.candidate_ids(t) for t in targets)))
            scores = {eid: float(rng.random()) for eid in ids if not entity or eid != entity.id}
            supports = {}
        elif method == "abc":
            scores, supports = self._abc(entity, targets)
        else:
            # Expand a known entity query with its aliases before retrieving papers.
            text = " ".join(dict.fromkeys([entity.name, *entity.aliases])) if entity else query
            ranked = self.retrieve(text, method)
            # Store one score per candidate/paper so repeated mentions do not double-count a paper.
            supports = defaultdict(dict)
            for i, score in ranked:
                doc = self.documents[i]
                # Retrieval must support the queried entity, not just another paper's similar topic.
                if entity and entity.id not in doc.entity_ids:
                    continue
                for eid in doc.entity_ids:
                    candidate = self.registry.entities[eid]
                    if candidate.kind in targets and self.candidate_allowed(candidate):
                        supports[eid][i] = score
            r = self.settings.retrieval
            # Step 3: collapse paper scores into one score per candidate entity.
            scores = {
                eid: aggregate(list(docs.values()), r.aggregation, r.aggregation_top_n)
                for eid, docs in supports.items()
            }
        if entity:
            scores.pop(entity.id, None)
        # Normalize each method's evidence scale before applying a common feasibility weight.
        # Step 4: put each method's entity scores on a within-query scale before BBB adjustments.
        high = max(scores.values(), default=0)
        low = min(scores.values(), default=0)
        results = []
        for eid, raw_score in scores.items():
            candidate = self.registry.entities[eid]
            # Preserve ordering when the supplied BM25/LM scores are all negative.
            # Raw reference scores remain available on each supporting Evidence record.
            if high > 0:
                base = raw_score / high
            elif high > low:
                base = (raw_score - low) / (high - low)
            else:
                base = 1.0
            adjustment = 0.0
            # Step 5: identify the drug/disease roles consistently in either search direction.
            if feasibility:
                drug = candidate if candidate.kind == "drug" else entity
                disease = entity if candidate.kind == "drug" else candidate
                if drug and disease and disease.cns:
                    adjustment = {"positive": 1, "negative": -1}.get(
                        drug.bbb, 0
                    ) * self.settings.retrieval.bbb_weight
            # Step 6: retain the strongest supporting papers for inspection, not generated
            # citations.
            doc_scores = sorted(supports.get(eid, {}).items(), key=lambda item: (-item[1], item[0]))
            evidence = [
                self._evidence(i, score, query)
                for i, score in doc_scores[: self.settings.retrieval.evidence_per_entity]
            ]
            results.append(
                SearchResult(
                    entity=candidate,
                    score=base + adjustment,
                    retrieval_score=base,
                    feasibility_adjustment=adjustment,
                    evidence_count=len(doc_scores),
                    evidence=evidence,
                )
            )
        # Final entity ties break by canonical ID to keep repeated searches deterministic.
        results.sort(key=lambda result: (-result.score, result.entity.id))
        if feasibility:
            warnings.append("BBB labels are retrospective; unknown/conflicting labels are neutral.")
        if entity and not self._entity_docs[entity.id]:
            warnings.append(
                "No indexed paper contains this query entity; literature coverage is missing."
            )
        return SearchResponse(
            query=query,
            direction=direction,
            method=method,
            resolved_entity=entity.id if entity else None,
            results=results[:k],
            warnings=warnings,
        )

    def _evidence(self, i, score, query):
        """Build a paper reference and choose a short query-overlap abstract sentence.

        Sentence selection uses token overlap with the original query, not a model
        extracting the candidate's therapeutic relationship. At most 700 characters
        are retained. The sentence may therefore be generic background and must be
        interpreted with the linked paper, not treated as a generated explanation.
        """
        doc = self.documents[i]
        # This simple sentence splitter/snippet chooser is independent of candidate relation
        # scoring.
        sentences = re.split(r"(?<=[.!?])\s+", doc.abstract)
        terms = set(tokenize(query))
        snippet = max(sentences, key=lambda s: len(set(tokenize(s)) & terms), default=doc.abstract)
        return Evidence(
            document_id=doc.id,
            title=doc.title,
            publication_date=doc.publication_date,
            url=doc.url,
            score=float(score),
            snippet=snippet[:700],
        )
