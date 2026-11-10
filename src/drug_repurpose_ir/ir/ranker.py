"""
This is the template for implementing the rankers for your search engine.
You will be implementing WordCountCosineSimilarity, DirichletLM, TF-IDF, BM25, Pivoted Normalization, and your own ranker.
"""
from collections import Counter, defaultdict
import numpy as np
from .indexing import InvertedIndex
from sentence_transformers import CrossEncoder
import math

class Ranker:
    """
    The ranker class is responsible for generating a list of documents for a given query, ordered by their scores
    using a particular relevance function (e.g., BM25).
    A Ranker can be configured with any RelevanceScorer.
    """
    # TODO: This class is responsible for returning a list of sorted relevant documents.

    def __init__(self, index: InvertedIndex, document_preprocessor, stopwords: set[str],
                 scorer: 'RelevanceScorer', raw_text_dict: dict[int, str] = None) -> None:
        """
        Initializes the state of the Ranker object.

        Args:
            index: An inverted index
            document_preprocessor: The DocumentPreprocessor to use for turning strings into tokens
            stopwords: The set of stopwords to use or None if no stopword filtering is to be done
            scorer: The RelevanceScorer object
            raw_text_dict: A dictionary mapping a document ID to the raw string of the document (Not needed for HW1)
        """
        self.index = index
        self.tokenize = document_preprocessor.tokenize
        self.scorer = scorer
        self.stopwords = stopwords
        self.raw_text_dict = raw_text_dict

    def query(self, query: str) -> list[tuple[int, float]]:
        """
        Searches the collection for relevant documents to the query and
        returns a list of documents ordered by their relevance (most relevant first).

        Args:
            query: The query to search for

        Returns:
            A sorted list containing tuples of the document id and its relevance score

        """
        # 1. Tokenize query (Hint: Also apply stopwords filtering to the tokenized query)

        # 2.1 For each token in the tokenized query, find out all documents that contain it and counting its frequency within each document.
        # Hint 1: To understand why we need the info above, pay attention to docid and doc_word_counts,
        #    located in the score() function within the RelevanceScorer class
        # Hint 2: defaultdict(Counter) works well in this case, where we store {docids : {query_tokens : counts}},
        #         or you may choose other approaches

        # 2.2 Run RelevanceScorer (like BM25 from below classes) (implemented as relevance classes)
        #        for each relevant document determined in 2.1

        # 3. Return **sorted** results as format [(docid_1, score), (docid_2, score)]

        tokens = self.tokenize(query)

        if self.stopwords is not None:
            tokens = [token for token in tokens if token not in self.stopwords]

        query_word_counts = Counter(tokens)

        doc_counts = defaultdict(Counter)

        for token in query_word_counts:
            document_list = self.index.get_postings(token)

            for tup in document_list:
                docid = tup[0]
                count = tup[1]

                doc_counts[docid][token] = count

        relevance_scores = []

        for docid, doc_word_counts in doc_counts.items():

            score = self.scorer.score(
                docid, doc_word_counts, query_word_counts)
            relevance_scores.append((docid, score))

        return sorted(relevance_scores, key=lambda x: x[1], reverse=True)


class RelevanceScorer:
    '''
    This is the base interface for all the relevance scoring algorithm.
    It will take a document and attempt to assign a score to it.
    '''
    # NOTE: Implement the functions in the child classes (WordCountCosineSimilarity, DirichletLM, BM25, PivotedNormalization, TF_IDF) and not in this one

    def __init__(self, index, parameters) -> None:
        raise NotImplementedError

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        """
        Returns a score for how relevance is the document for the provided query.

        Args:
            docid: The ID of the document
            doc_word_counts: A dictionary containing all words in the document that are also found in the query, 
                                and their frequencies within the document.
                Words that have been filtered will be None.
            query_word_counts: A dictionary containing all words in the query and their frequencies.
                Words that have been filtered will be None.

        Returns:
            A score for how relevant the document is (Higher scores are more relevant.)

        """
        raise NotImplementedError


class SampleScorer(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters) -> None:
        pass

    def score(self, docid: int, doc_word_counts: dict[str, int], query_parts: list[str]) -> float:
        """
        Scores all documents as 10.
        """
        return 10


class WordCountCosineSimilarity(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters: dict = {}) -> None:
        self.index = index
        self.parameters = parameters

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        # 1. Find the dot product of the word count vector of the document and the word count vector of the query

        # 2. Return the score
        final_score = 0
        for token in query_word_counts:
            final_score = final_score + \
                doc_word_counts[token] * query_word_counts[token]
        return final_score


class DirichletLM(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters: dict = {'mu': 2000}) -> None:
        self.index = index
        self.parameters = parameters

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        # 1. Get necessary information from index

        # 2. Compute additional terms to use in algorithm

        # 3. For all query_parts, compute score

        # 4. Return the score
        final_score = 0

        mu = self.parameters["mu"]
        collection_count = self.index.get_statistics()[
            'total_token_count']
        q = query_word_counts.total()

        for token in query_word_counts:

            cq = query_word_counts[token]

            first = cq

            cd = doc_word_counts[token]
            term_count = self.index.get_term_metadata(token)['term_count']
            p = term_count / collection_count

            second = math.log(1 + (cd) / (mu * p))

            final_score = final_score + first * second

        d = self.index.get_doc_metadata(docid)['length']
        third = q * math.log(mu / (d + mu))

        final_score = final_score + third
        return final_score


class BM25(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters: dict = {'b': 0.75, 'k1': 1.2, 'k3': 8}) -> None:
        self.index = index
        self.b = parameters['b']
        self.k1 = parameters['k1']
        self.k3 = parameters['k3']

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        # 1. Get necessary information from index

        # 2. Find the dot product of the word count vector of the document and the word count vector of the query

        # 3. For all query parts, compute the TF and IDF to get a score

        # 4. Return score
        final_score = 0

        N = self.index.get_statistics()['number_of_documents']
        avdl = self.index.get_statistics()['mean_document_length']
        mod_d = self.index.get_doc_metadata(docid)['length']

        for token in query_word_counts:

            df = self.index.get_term_metadata(token)['doc_frequency']
            first = math.log((N - df + 0.5) / (df + 0.5))

            cd = doc_word_counts[token]
            second = ((self.k1 + 1) * cd) / (self.k1 *
                                             (1 - self.b + self.b * (mod_d / avdl)) + cd)

            cq = query_word_counts[token]
            third = ((self.k3 + 1) * cq) / (self.k3 + cq)

            final_score = final_score + first * second * third

        return final_score


class PivotedNormalization(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters: dict = {'b': 0.2}) -> None:
        self.index = index
        self.b = parameters['b']

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        # 1. Get necessary information from index

        # 2. Compute additional terms to use in algorithm

        # 3. For all query parts, compute the TF, IDF, and QTF values to get a score

        # 4. Return the score
        final_score = 0

        D = self.index.get_statistics()["number_of_documents"]
        avdl = self.index.get_statistics()['mean_document_length']
        mod_d = self.index.get_doc_metadata(docid)['length']

        for token in query_word_counts:
            cq = query_word_counts[token]
            cd = doc_word_counts[token]

            if cd == 0:
                continue

            first = cq

            second_n = 1 + math.log(1 + math.log(cd))
            second_d = 1 - self.b + self.b * (mod_d / avdl)
            second = second_n / second_d

            df = self.index.get_term_metadata(token)['doc_frequency']
            third = math.log((D + 1) / df)

            final_score = final_score + first * second * third

        return final_score


class TF_IDF(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters: dict = {}) -> None:
        self.index = index
        self.parameters = parameters

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        # 1. Get necessary information from index

        # 2. Compute additional terms to use in algorithm

        # 3. For all query parts, compute the TF and IDF to get a score

        # 4. Return the score
        final_score = 0

        D = self.index.get_statistics()["number_of_documents"]

        for token in query_word_counts:
            cd = doc_word_counts[token]
            df = len(self.index.get_postings(token))

            first = math.log(cd + 1)
            second = math.log(D/df) + 1

            final_score = final_score + first * second

        return final_score

class YourRanker(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters: dict = {"alpha": 0.5}) -> None:
        self.index = index
        self.parameters = parameters

    def score(self, docid: int, doc_word_counts: dict[str, int],
              query_word_counts: dict[str, int]) -> float:

        score = 0
        alpha = self.parameters["alpha"]

        for token in query_word_counts:
            score += doc_word_counts[token] * query_word_counts[token]

        document_length = self.index.get_doc_metadata(docid)['length']

        score = score / document_length ** alpha

        return score

# TODO (HW3): The CrossEncoderScorer class uses a pre-trained cross-encoder model from the Sentence Transformers package
#             to score a given query-document pair; check README for details
#
# NOTE: This is not a RelevanceScorer object because the method signature for score() does not match, but it
# has the same intent, in practice
class CrossEncoderScorer:
    '''
    A scoring object that uses cross-encoder to compute the relevance of a document for a query.
    '''

    def __init__(self, raw_text_dict: dict[int, str],
                 cross_encoder_model_name: str = 'cross-encoder/msmarco-MiniLM-L6-en-de-v1') -> None:
        """
        Initializes a CrossEncoderScorer object.
            Visit https://sbert.net/docs/cross_encoder/usage/usage.html for more information.

        Args:
            raw_text_dict: A dictionary where the document id is mapped to a string with the first 500 words
                in the document
            cross_encoder_model_name: The name of a cross-encoder model
        """
        # TODO (HW3): Save any new arguments that are needed as fields of this class
        self.model = CrossEncoder(cross_encoder_model_name)
        self.raw_text_dict = raw_text_dict

    def score(self, docid: int, query: str) -> float:
        """
        Gets the cross-encoder score for the given document.

        Args:
            docid: The id of the document
            query: The query in its original form (no stopword filtering/tokenization)

        Returns:
            The score returned by the cross-encoder model
        """
        # NOTE: unlike the other scorers like BM25, this method takes in the query string itself,
        # not the tokens!

        # TODO (HW3): Do not forget to handle an edge case
        # (e.g., docid does not exist in raw_text_dict or empty query, both should lead to 0 score)
        if not query or not query.strip() or docid not in self.raw_text_dict:
            return 0


        # TODO (HW3): Get a score from the cross-encoder model
        #             Refer to IR_Encoder_Examples.ipynb in Demos folder on Canvas if needed
        doc = self.raw_text_dict[docid]
        pairs = [(query, doc)]

        scores = self.model.predict(pairs)
        return scores[0]