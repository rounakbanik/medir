"""Convert project records to the interfaces used by the reference IR code.

The implementations live in ir/. This file handles IDs, model configuration,
and short evaluation lists; it does not reimplement the reference algorithms.
"""

from pathlib import Path

from .io import digest, write_jsonl

LEXICAL_METHODS = {"bm25", "tf-idf", "dirichlet", "pivoted", "word-count", "custom"}


def implementation_signature():
    """Include the actual packaged reference code in index compatibility checks."""
    folder = Path(__file__).parent / "ir"
    return {p.name: digest(p) for p in sorted(folder.glob("*.py"))}


def reference_tokenizer(settings=None):
    """Use the reference RegexTokenizer for both documents and lexical queries."""
    from .ir.document_preprocessor import RegexTokenizer

    pattern = settings.retrieval.token_regex if settings else r"\w+"
    return RegexTokenizer(pattern)


def build_reference_index(settings, documents):
    """Export JSONL with integer docids, then call the reference Indexer and save().

    Integer IDs are row numbers in documents.jsonl. The original PubMed/local IDs
    remain in that file, so a ranking can be mapped back without changing the
    reference index's save/load format.
    """
    from .ir.indexing import Indexer, IndexType

    target = settings.paths.index
    corpus = target / "corpus.jsonl"
    write_jsonl(
        corpus,
        (
            {"docid": i, "title": doc.title, "text": doc.text + " " + " ".join(doc.mesh)}
            for i, doc in enumerate(documents)
        ),
    )
    index_type = (
        IndexType.PositionalIndex
        if settings.retrieval.index_type == "positional"
        else IndexType.BasicInvertedIndex
    )
    index = Indexer.create_index(
        index_type,
        str(corpus),
        reference_tokenizer(settings),
        set(settings.retrieval.stopwords),
        settings.retrieval.minimum_word_frequency,
    )
    index.save(str(target / "lexical"))
    return index


def load_reference_index(settings):
    """Restore the reference basic or positional index from its own JSON format."""
    from .ir.indexing import BasicInvertedIndex, PositionalInvertedIndex

    index = (
        PositionalInvertedIndex()
        if settings.retrieval.index_type == "positional"
        else BasicInvertedIndex()
    )
    index.load(str(settings.paths.index / "lexical"))
    return index


def reference_ranker(index, settings, method="bm25"):
    """Configure one of the supplied scorers and the supplied Ranker around it."""
    from .ir.ranker import (
        BM25,
        TF_IDF,
        DirichletLM,
        PivotedNormalization,
        Ranker,
        WordCountCosineSimilarity,
        YourRanker,
    )

    r = settings.retrieval
    scorers = {
        "bm25": lambda: BM25(index, {"b": r.bm25_b, "k1": r.bm25_k1, "k3": r.bm25_k3}),
        "tf-idf": lambda: TF_IDF(index),
        "dirichlet": lambda: DirichletLM(index, {"mu": r.dirichlet_mu}),
        "pivoted": lambda: PivotedNormalization(index, {"b": r.pivoted_b}),
        "word-count": lambda: WordCountCosineSimilarity(index),
        "custom": lambda: YourRanker(index, {"alpha": r.custom_alpha}),
    }
    return Ranker(index, reference_tokenizer(settings), set(r.stopwords), scorers[method]())


def lexical_query(ranker, query, method):
    """Pass text to Ranker.query, omitting OOV terms for TF-IDF and DirichletLM.

    TF-IDF divides by document frequency; DirichletLM divides by collection
    probability. Their original implementations require query terms to occur in
    the collection. Filter them here without changing the supplied score().
    """
    if method in {"dirichlet", "tf-idf"}:
        query = " ".join(t for t in ranker.tokenize(query) if t in ranker.index.vocabulary)
    return ranker.query(query)


def reference_vector_ranker(settings, embeddings, revision=None):
    """Construct the supplied VectorRanker using the document model's snapshot."""
    from .indexing import neural_environment

    neural_environment(settings)
    from huggingface_hub import try_to_load_from_cache

    from .ir.vector_ranker import VectorRanker

    r = settings.retrieval
    model_path = r.dense_model
    if revision and not Path(model_path).is_dir():
        cached = try_to_load_from_cache(model_path, "config.json", revision=revision)
        if not isinstance(cached, str):
            raise ValueError("The embedding model snapshot is not cached; run index --dense")
        model_path = str(Path(cached).parent)
    ranker = VectorRanker(model_path, embeddings, list(range(len(embeddings))))
    ranker.biencoder_model.to(r.device)
    ranker.biencoder_model.max_seq_length = r.max_sequence_length
    return ranker


def reference_cross_encoder(settings, documents):
    """Provide the supplied CrossEncoderScorer its expected integer-ID/raw-text map."""
    from .indexing import neural_environment

    neural_environment(settings)
    from .ir.ranker import CrossEncoderScorer

    r = settings.retrieval
    model_path = r.cross_encoder_model
    if r.cross_encoder_revision and not Path(model_path).is_dir():
        from huggingface_hub import snapshot_download

        model_path = snapshot_download(model_path, revision=r.cross_encoder_revision)
    texts = {i: " ".join(doc.text.split()[:500]) for i, doc in enumerate(documents)}
    scorer = CrossEncoderScorer(texts, model_path)
    scorer.model.model.to(r.device)
    scorer.model.max_length = r.max_sequence_length
    return scorer


def reference_metrics(ranked_relevances, ideal_relevances, k):
    """Call the original MAP/nDCG functions, padding only short nDCG inputs.

    MAP deliberately retains the reference denominator: the positives in the
    supplied ranking, capped at k. Recall elsewhere still uses all judgments.
    No unseen positives are fabricated or appended to the ranked list.
    """
    from .ir.relevance import map_score, ndcg_score

    ranked = list(ranked_relevances)
    ideal = list(ideal_relevances)
    map_value = map_score(ranked, cut_off=k)
    ranked.extend([0] * max(0, k - len(ranked)))
    ideal.extend([0] * max(0, k - len(ideal)))
    return float(map_value), float(ndcg_score(ranked, ideal, cut_off=k))
