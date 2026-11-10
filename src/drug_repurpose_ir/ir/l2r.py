import pandas as pd
import lightgbm
from .indexing import InvertedIndex
from .document_preprocessor import Tokenizer
from .ranker import *
from collections import defaultdict, Counter

# TODO (HW3): CrossEncoderScorer has been added to __init__ of L2RFeatureExtractor
# Make sure to update your code accordingly.
class L2RFeatureExtractor:
    def __init__(self, document_index: InvertedIndex, title_index: InvertedIndex,
                 doc_category_info: dict[int, list[str]],
                 document_preprocessor: Tokenizer, stopwords: set[str],
                 recognized_categories: set[str], docid_to_network_features: dict[int, dict[str, float]],
                 ce_scorer: CrossEncoderScorer) -> None:
        """
        Initializes a L2RFeatureExtractor object.

        Args:
            document_index: The inverted index for the contents of the document's main text body
            title_index: The inverted index for the contents of the document's title
            doc_category_info: A dictionary where the document id is mapped to a list of categories
            document_preprocessor: The DocumentPreprocessor to use for turning strings into tokens
            stopwords: The set of stopwords to use or None if no stopword filtering is to be done
            recognized_categories: The set of categories to be recognized as binary features
                (whether the document has each one)
            docid_to_network_features: A dictionary where the document id is mapped to a dictionary
                with keys for network feature names "PageRank", "Hub", and "Authority"
                and values with the scores for those features
                dict[int, dict[str, float]]: dict{docid: {network feature names: values}}
            ce_scorer: The CrossEncoderScorer object
        """
        self.document_index = document_index
        self.title_index = title_index
        self.doc_category_info = doc_category_info
        self.document_preprocessor = document_preprocessor
        self.stopwords = stopwords
        self.recognized_categories = sorted(recognized_categories)
        self.docid_to_network_features = docid_to_network_features

        # TODO: Set the initial state using the arguments
        self.ce_scorer = ce_scorer


    def get_article_length(self, docid: int) -> int:
        """
        Gets the length of a document (including stopwords).

        Args:
            docid: The id of the document

        Returns:
            The length of a document
        """
        return self.document_index.get_doc_metadata(docid)["length"]


    def get_title_length(self, docid: int) -> int:
        """
        Gets the length of a document's title (including stopwords).

        Args:
            docid: The id of the document

        Returns:
            The length of a document's title
        """
        return self.title_index.get_doc_metadata(docid)["length"]


    def get_tf(self, index: InvertedIndex, docid: int, word_counts: dict[str, int], query_parts: list[str]) -> float:
        """
        Calculates the TF score: for w in q and d, sum(lg(c(w) + 1))

        Args:
            index: An inverted index to use for calculating the statistics
            docid: The id of the document
            word_counts: The words in some part of a document mapped to their frequencies
            query_parts: A list of tokenized query tokens

        Returns:
            The TF score
        """
        tf = 0

        for word in set(query_parts):
            cd = word_counts.get(word, 0)
            tf += math.log(cd + 1)
        
        return tf


    def get_tf_idf(self, index: InvertedIndex, docid: int,
                    word_counts: dict[str, int], query_parts: list[str]) -> float:
        """
        Calculates the TF-IDF score.

        Args:
            index: An inverted index to use for calculating the statistics
            docid: The id of the document
            word_counts: The words in some part of a document mapped to their frequencies
            query_parts: A list of tokenized query tokens

        Returns:
            The TF-IDF score
        """
        scorer = TF_IDF(index)
        doc_word_counts = Counter(word_counts)
        query_word_counts = Counter(
            word for word in query_parts
            if index.get_term_metadata(word)["doc_frequency"] > 0
        )

        return scorer.score(docid, doc_word_counts, query_word_counts)
    

    def get_BM25_score(self, docid: int, doc_word_counts: dict[str, int],
                       query_parts: list[str]) -> float:
        """
        Calculates the BM25 score.

        Args:
            docid: The id of the document
            doc_word_counts: The words in the document's main text mapped to their frequencies
            query_parts: A list of tokenized query tokens

        Returns:
            The BM25 score
        """
        # NOTE: Use the default parameters of BM25 ranker
        scorer = BM25(self.document_index)
        doc_word_counts = Counter(doc_word_counts)
        query_word_counts = Counter(query_parts)

        return scorer.score(docid, doc_word_counts, query_word_counts)
        

    def get_pivoted_normalization_score(self, docid: int, doc_word_counts: dict[str, int],
                                        query_parts: list[str]) -> float:
        """
        Calculates the pivoted normalization score.

        Args:
            docid: The id of the document
            doc_word_counts: The words in the document's main text mapped to their frequencies
            query_parts: A list of tokenized query tokens

        Returns:
            The pivoted normalization score
        """
        # NOTE: Use the default parameters of PivotedNormalization ranker
        scorer = PivotedNormalization(self.document_index)
        doc_word_counts = Counter(doc_word_counts)
        query_word_counts = Counter(query_parts)

        return scorer.score(docid, doc_word_counts, query_word_counts)


    def get_document_categories(self, docid: int) -> list:
        """
        Generates a list of binary features indicating which of the recognized categories that the document has.
        Category features should be deterministically ordered so list[0] should always correspond to the same
        category. For example, if a document has one of the three categories, and that category is mapped to
        index 1, then the binary feature vector would look like [0, 1, 0].

        Args:
            docid: The id of the document

        Returns:
            A list containing binary list of which recognized categories that the given document has
        """
        doc_categories = self.doc_category_info.get(docid, [])

        binary_list = [
            0 if category not in doc_categories else 1 
            for category in self.recognized_categories
        ]
        return binary_list



    def get_pagerank_score(self, docid: int) -> float:
        """
        Gets the PageRank score for the given document.

        Args:
            docid: The id of the document

        Returns:
            The PageRank score if docid is in the network features, otherwise 0
        """
        return self.docid_to_network_features.get(docid, {}).get("pagerank", 0)


    def get_hits_hub_score(self, docid: int) -> float:
        """
        Gets the HITS hub score for the given document.

        Args:
            docid: The id of the document

        Returns:
            The HITS hub score if docid is in the network features, otherwise 0
        """
        return self.docid_to_network_features.get(docid, {}).get("hub_score", 0)

    
    def get_hits_authority_score(self, docid: int) -> float:
        """
        Gets the HITS authority score for the given document.

        Args:
            docid: The id of the document

        Returns:
            The HITS authority score if docid is in the network features, otherwise 0
        """
        return self.docid_to_network_features.get(docid, {}).get("authority_score", 0)
    
    # TODO (HW3): Cross-Encoder Score
    def get_cross_encoder_score(self, docid: int, query: str) -> float:
        """
        Gets the cross-encoder score for the given document.

        Args:
            docid: The id of the document
            query: The query in its original form (no stopword filtering/tokenization)

        Returns:
            The Cross-Encoder score
        """        
        return self.ce_scorer.score(docid, query)


    def get_my_feature(self, docid: int) -> float:
        """
        Calculates your new feature.

        Args:
            docid: The id of the document
            ... add more arguments if you need them

        Returns:
            The value of your new feature
        """
        metadata = self.document_index.get_doc_metadata(docid)

        unique_words = metadata["unique_tokens"]
        total_words = metadata["length"]

        if total_words == 0:
            return 0.0
        
        return unique_words / total_words

    # TODO (HW3): generate_features() has been updated to include the query in its original form to get the cross-encoder score.
    # Make sure to update your code accordingly.
    def generate_features(self, docid: int, doc_word_counts: dict[str, int],
                          title_word_counts: dict[str, int], query_parts: list[str],
                          query: str) -> list:
        """
        Generates a dictionary of features for a given document and query.

        Args:
            docid: The id of the document to generate features for
            doc_word_counts: The words in the document's main text mapped to their frequencies
            title_word_counts: The words in the document's title mapped to their frequencies
            query_parts : A list of tokenized query terms to generate features for
            query: The original query string

        Returns:
            A vector (list) of the features for this document
                Feature order should be stable between calls to the function
                (the order of features in the vector should not change).
        """
        feature_vector = []

        # Document Length
        feature_vector.append(self.get_article_length(docid))

        # Title Length
        feature_vector.append(self.get_title_length(docid))

        # Query Length
        feature_vector.append(len(query_parts))

        # TF (document)
        feature_vector.append(
            self.get_tf(self.document_index, docid, doc_word_counts, query_parts)
        )

        # TF-IDF (document)
        feature_vector.append(
            self.get_tf_idf(self.document_index, docid, doc_word_counts, query_parts)
        )

        # TF (title)
        feature_vector.append(
            self.get_tf(self.title_index, docid, title_word_counts, query_parts)
        )

        # TF-IDF (title)
        feature_vector.append(
            self.get_tf_idf(self.title_index, docid, title_word_counts, query_parts)
        )

        # BM25
        feature_vector.append(
            self.get_BM25_score(docid, doc_word_counts, query_parts)
        )

        # Pivoted Normalization
        feature_vector.append(
            self.get_pivoted_normalization_score(docid, doc_word_counts, query_parts)
        )

        # Pagerank
        feature_vector.append(self.get_pagerank_score(docid))

        # HITS Hub
        feature_vector.append(self.get_hits_hub_score(docid))

        # HITS Authority
        feature_vector.append(self.get_hits_authority_score(docid))

        # TODO (HW3): Cross-Encoder Score
        feature_vector.append(self.get_cross_encoder_score(docid, query))

        # Add at least one new feature to be used with your L2R model.
        feature_vector.append(
            self.get_my_feature(docid)
        )

        # Calculate the Document Categories features.
        # This should be a list of binary values indicating which categories are present.
        feature_vector.extend(self.get_document_categories(docid))

        return feature_vector
        

class LambdaMART:
    def __init__(self, params=None) -> None:
        """
        LightGBM is a gradient boosting framework that uses tree based learning algorithms
        To learn more, visit: https://lightgbm.readthedocs.io/en/stable/ and https://lightgbm.readthedocs.io/en/stable/Python-Intro.html

        If lightgbm is not installed, you can install it via pip:
        It is recommended to install packages in a virtual environment.
            Here is an example to do so:
                $ python -m venv [your python virtual enviroment]
                $ source [your python virtual enviroment]/bin/activate # or [your python virtual environment]\Scripts\activate on Windows
                $ pip install lightgbm

        Your task:
        Initializes a LambdaMART (LGBRanker) model using the lightgbm library.

        Args:
            params (dict, optional): Parameters for the LGBMRanker model. Defaults to None.
        """
        default_params = {
            'objective': "lambdarank",
            'boosting_type': "gbdt",
            'n_estimators': 10,
            'importance_type': "gain",
            'metric': "ndcg",
            'num_leaves': 20,
            'learning_rate': 0.04,
            'max_depth': -1,
            # NOTE: You might consider setting this parameter to a higher value equal to
            # the number of CPUs on your machine for faster training
            "n_jobs": 1,
        }

        if params:
            default_params.update(params)

        # Initialize the LGBMRanker with the provided parameters and assign as a field of this class
        self.model = lightgbm.LGBMRanker(**default_params)
        

    def fit(self, X_train, y_train, qgroups_train):
        """
        Trains the LGBMRanker model.

        Args:
            X_train (array-like): Training input samples.
            y_train (array-like): Target values.
            qgroups_train (array-like): Query group sizes for training data.

        Returns:
            self: Returns the instance itself.
        """
        
        # fit the LGBMRanker's parameters using the provided features and labels
        # Visit https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRanker.html#lightgbm.LGBMRanker.fit for details
        return self.model.fit(X_train, y_train, group=qgroups_train)

    def predict(self, featurized_docs):
        """
        Predicts the relevance (ranks) of the featurized documents and return the values.

        Args:
            featurized_docs (array-like): 
                A list of featurized documents where each document is a list of its features
                All documents should have the same length.

        Returns:
            array-like: The estimated ranking for each document (unsorted)
        """

        # Generating the predicted values using the LGBMRanker
        # Visit https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRanker.html#lightgbm.LGBMRanker.predict for details
        return self.model.predict(featurized_docs)
    
# TODO (HW3): __init__ of L2RRanker has been updated. In HW2, L2RRanker took a Scorer object as input.
# In HW3, L2RRanker takes a Ranker object as input. Make sure to update your code accordingly.
class L2RRanker:
    def __init__(self, document_index: InvertedIndex, title_index: InvertedIndex,
                 document_preprocessor: Tokenizer, stopwords: set[str], ranker: Ranker,
                 feature_extractor: 'L2RFeatureExtractor') -> None:
        """
        Initializes a L2RRanker model.

        Args:
            document_index: The inverted index for the contents of the document's main text body
            title_index: The inverted index for the contents of the document's title
            document_preprocessor: The DocumentPreprocessor to use for turning strings into tokens
            stopwords: The set of stopwords to use or None if no stopword filtering is to be done
            ranker: The Ranker object ** HW3 MODIFIED **
            feature_extractor: The L2RFeatureExtractor object
        """
        # Save any arguments that are needed as fields of this class
        self.document_index = document_index
        self.title_index = title_index
        self.document_preprocessor = document_preprocessor
        self.stopwords = stopwords
        self.ranker = ranker
        self.feature_extractor = feature_extractor


        # Initialize the LambdaMART model (but don't train it yet)
        self.model = LambdaMART() # This should a LambdaMART object (defined in this script)


    def prepare_training_data(self, query_to_document_relevance_scores: dict[str, list[tuple[int, int]]]) -> tuple[list, list, list]:
        """
        Prepares the training data for the learning-to-rank algorithm.

        Args:
            query_to_document_relevance_scores (dict): A dictionary of queries mapped to a list of 
                documents and their relevance scores for that query
                The dictionary has the following structure:
                {
                    query_1_text: [(docid_1, relevanceScore_to_query_1), (docid_2, relevanceScore_to_query_1), ...],
                    query_2_text: [(docid_3, relevanceScore_to_query_2), (docid_4, relevanceScore_to_query_2), ...],
                    ...
                }

        Returns:
            tuple: A tuple containing the training data in the form of three lists: X, y, and qgroups
                X (list): A list of feature vectors for each query-document pair
                y (list): A list of relevance scores for each query-document pair
                qgroups (list): A list of the number of documents retrieved for each query
        """
        # NOTE: qgroups is not the same length as X or y.
        # This is for LightGBM to know how many relevance scores we have per query.
        X = []
        y = []
        qgroups = []

        for query, documents in query_to_document_relevance_scores.items():
            
            # DONE: for each query and the documents that have been rated for relevance to that query,
            # process these query-document pairs into features
            query_parts = self.document_preprocessor.tokenize(query)

            if self.stopwords is not None:
                query_parts = [token for token in query_parts if token not in self.stopwords]
            
            
            # DONE: Accumulate the token counts for each document's title and content here
            doc_counts = self.accumulate_doc_term_counts(self.document_index, query_parts)
            title_counts = self.accumulate_doc_term_counts(self.title_index, query_parts)

            # DONE: For each of the documents, generate its features, then append
            # the features and relevance score to the lists to be returned
            for docid, score in documents:
                features = self.feature_extractor.generate_features(
                    docid,
                    doc_counts.get(docid, {}),
                    title_counts.get(docid, {}),
                    query_parts,
                    query,
                )

                X.append(features)
                y.append(score)

            # DONE: Make sure to keep track of how many scores we have for this query in qrels
            qgroups.append(len(documents))

        return X, y, qgroups

    @staticmethod
    def accumulate_doc_term_counts(index: InvertedIndex, query_parts: list[str]) -> dict[int, dict[str, int]]:
        """
        A helper function that for a given query, retrieves all documents that have any
        of these words in the provided index and returns a dictionary mapping each document id to
        the counts of how many times each of the query words occurred in the document
        
        Args:
            index: An inverted index to search
            query_parts: A list of tokenized query tokens

        Returns:
            A dictionary mapping each document containing at least one of the query tokens to
            a dictionary with how many times each of the query words appears in that document
            
        """
        # Retrieve the set of documents that have each query word (i.e., the postings) and
        # create a dictionary that keeps track of their counts for the query word
        doc_counts = defaultdict(dict)

        for token in set(query_parts):
            document_list = index.get_postings(token)

            for tup in document_list:
                docid = tup[0]
                count = tup[1]

                doc_counts[docid][token] = count

        return dict(doc_counts)
        
    def train(self, training_data_filename: str) -> None:
        """
        Trains a LambdaMART pair-wise learning to rank model using the documents and relevance scores provided 
        in the training data file.

        Args:
            training_data_filename (str): a filename for a csv file containing documents and relevance scores
        """
        # Convert the relevance data into the right format for training data preparation

        # Prepare the training data by featurizing the query-doc pairs and

        # Train the model
        try:
            df = pd.read_csv(training_data_filename)
        except:
            df = pd.read_csv(training_data_filename, encoding="cp850")


        # DONE: prepare the training data by featurizing the query-doc pairs and
        # getting the necessary datastructures
        training_data = {}

        for query, cols in df.groupby("query", sort=False):
            training_data[query] = list(zip(cols["docid"], cols["rel"]))
        
        X, y, qgroups = self.prepare_training_data(training_data)

        # DONE: Train the model
        self.model.fit(X, y, qgroups)
        
    def predict(self, X):
        """
        Predicts the ranks for featurized doc-query pairs using the trained model.

        Args:
            X (array-like): Input data to be predicted
                This is already featurized doc-query pairs.

        Returns:
            array-like: The predicted rank of each document

        Raises:
            ValueError: If the model has not been trained yet.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        # Return a prediction made using the LambdaMART model
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        # DONE: Return a prediction made using the LambdaMART model
        return self.model.predict(X)

    def query(self, query: str) -> list[tuple[int, float]]:
        """
        Summary: 
            Retrieves potentially-relevant documents, constructs feature vectors for each query-document pair,
            uses the L2R model to rank these documents, and returns the ranked documents.
        
        Description:
            Retrieves potentially-relevant documents using the RelevanceScorer in __init__,
            Seperates the top 100 most potentially-relevant documents and the remaining documents, if any.
            Constructs feature vectors for each query-document pair in the top 100.
            Uses the L2R model to rank these top 100 documents. Combines the re-ranked top 100 documents 
                with the remaining documents, and returns the ranked documents.

        Args:
            query: A string representing the query to be used for ranking

        Returns:
            A list containing tuples of the ranked documents and their scores, sorted by score in descending order
                The list has the following structure: [(doc_id_1, score_1), (doc_id_2, score_2), ...]
        """
        # Return an empty list if the query is empty, else tokenize the query using the document preprocessor and stopword filter
        if not query:
            return []
        
        query_parts = self.document_preprocessor.tokenize(query)

        if self.stopwords is not None:
            query_parts = [
                token for token in query_parts
                if token not in self.stopwords
            ]

        # Accumulate the potentially-relevant documents word frequencies for the title and the main body
        doc_counts = self.accumulate_doc_term_counts(self.document_index, query_parts)
        title_counts = self.accumulate_doc_term_counts(self.title_index, query_parts)


        # TODO (HW3): Score and sort the documents by the provided RANKER (SCORER in HW2) for just the document's main text (not the title)
        # This ordering determines which documents we will try to *re-rank* using our L2R model

        # NOTE (HW3): __init__ of L2RRanker has been updated. In HW2, L2RRanker took a Scorer object as input.
        # In HW3, L2RRanker takes a Ranker object as input. Make sure to update your code accordingly.
        base_scores = self.ranker.query(query)

        # If no potentially-relevant documents are found, return an empty list
        if not base_scores:
            return []

        # Filter to just the top 100 documents for the L2R part for re-ranking
        top = base_scores[:100]
        bottom = base_scores[100:]

        # Construct the feature vectors for each query-document pair in the top 100
        X = []

        for docid, score in top:
            features = self.feature_extractor.generate_features(
                docid,
                doc_counts.get(docid, {}),
                title_counts.get(docid, {}),
                query_parts,
                query,
            )

            X.append(features)

        # Use your L2R model to rank these top 100 documents
        y_pred = self.predict(X)

        # Sort posting_lists based on scores
        ranked = []

        for doc, score in zip(top, y_pred):
            ranked.append((doc[0], score))

        ranked = sorted(ranked, key=lambda x: x[1], reverse=True)
        
        # Make sure to add back the other non-top-100 documents that weren't re-ranked
        ranked.extend(bottom)

        # Return the ranked documents
        return ranked