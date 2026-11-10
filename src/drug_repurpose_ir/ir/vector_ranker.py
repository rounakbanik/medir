from sentence_transformers import SentenceTransformer, util
import numpy as np
from .ranker import Ranker

class VectorRanker(Ranker):
    def __init__(self, bi_encoder_model_name: str, encoded_docs: np.ndarray,
                 row_to_docid: list[int]) -> None:
        """
        Requirements:
            Please install sentence-transformers to your virtual environment.
                $ pip install -U sentence-transformers
            
            You can find the documentation here: https://sbert.net/docs/sentence_transformer/usage/usage.html

        Overview:
            Initializes a VectorRanker object.

        Args:
            bi_encoder_model_name: The name of a huggingface model to use for initializing a 'SentenceTransformer'
            encoded_docs: A matrix where each row is an already-encoded document, encoded using the same encoding method
                as specified with bi_encoded_model_name
            row_to_docid: A list that is a mapping from the row number to the document id that row corresponds to
                the embedding

        Using zip(encoded_docs, row_to_docid) should give you the mapping between the docid and the embedding.
        """
        # DONE (HW3): Instantiate the bi-encoder model here

        # NOTE (HW3): we're going to use the cpu for everything here so if you decide to use a GPU, do not 
        # submit that code to the autograder
        self.biencoder_model = SentenceTransformer(bi_encoder_model_name) # Initialize the bi-encoder model here

        # DONE (HW3): Also include other necessary initialization code
        self.encoded_docs = np.asarray(encoded_docs)
        self.row_to_docid = row_to_docid


    def query(self, query: str) -> list[tuple[int, float]]:
        """
        Encodes the query and then scores the relevance of the query with all the documents.

        Args:
            query: The query in its original form (no stopword filtering/tokenization)

        Returns:
            A sorted list of tuples containing the document id and its relevance to the query,
            with most relevant documents first or an empty list if a query cannot be encoded
            or no results are return
        """
        # TODO (HW3): Return empty list if query is empty
        if not query or not query.strip():
            return []

        
        # TODO (HW3): Encode the query using the bi-encoder
        q_embedding = self.biencoder_model.encode(query)

        # TODO (HW3): Score the similarity of the query vector and document vectors for relevance
        # Calculate the dot products between the query embedding and all document embeddings
        # NOTE: Refer to IR_Encoder_Examples.ipynb on Canvas if needed
        scores = util.dot_score(
            q_embedding,
            self.encoded_docs,
        )[0].cpu().tolist()

        # TODO (HW3): Generate the ordered list of (document id, score) tuples
        doc_score_pairs = list(zip(self.row_to_docid, scores))

        # TODO (HW3): Sort the list so most relevant are first
        doc_score_pairs = sorted(
            doc_score_pairs,
            key=lambda x: x[1],
            reverse=True
        )

        # TODO (HW3): Return the sorted list
        return doc_score_pairs