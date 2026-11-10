'''
Here you will be implemeting the indexing strategies for your search engine. You will need to create, persist and load the index.
This will require some amount of file handling.
DO NOT use the pickle module.
'''
import os
import gzip
import json
from enum import Enum
from collections import Counter, defaultdict
from tqdm import tqdm
from .document_preprocessor import Tokenizer


class IndexType(Enum):
    # the two types of index currently supported are BasicInvertedIndex and PositionalIndex
    BasicInvertedIndex = 'BasicInvertedIndex'
    PositionalIndex = 'PositionalIndex'
    SampleIndex = 'SampleIndex'


class InvertedIndex:
    def __init__(self) -> None:
        """
        The base interface representing the data structure for all index classes.
        The functions are meant to be implemented in the actual index classes and not as part of this interface.

        Note: The following variables are defined to help you store some summary info about your document collection
                for a quick look-up.
              You may also define more variables and/or keys as you see fit.
        Variables:
            statistics: A dictionary, which is the central statistics of the index.
                        Some keys include:
                statistics['vocab']: A counter which keeps track of the token count
                statistics['unique_token_count']: how many unique terms are in the index
                statistics['total_token_count']: how many total tokens are indexed including filterd tokens),
                    i.e., the sum of the lengths of all documents
                statistics['stored_total_token_count']: how many total tokens are indexed excluding filtered tokens
                statistics['number_of_documents']: the number of documents indexed
                statistics['mean_document_length']: the mean number of tokens in a document (including filter tokens)
                ...
                (Add more keys to the statistics dictionary as you see fit)

            vocabulary: A set of distinct words that have appeared in the collection
            document_metadata: A dictionary, which keeps track of some important metadata for each document.
                               Assume that we have a document called 'doc1', some keys include:
                document_metadata['doc1']['unique_tokens']: How many unique tokens are in the document (among those not-filtered)
                document_metadata['doc1']['length']: How long the document is in terms of tokens (including those filtered)
                ...
                (Add more keys to the document_metadata dictionary as you see fit)
            index: A dictionary of class defaultdict, its implemention depends on whether we are using
                            BasicInvertedIndex or PositionalIndex.
                    BasicInvertedIndex: Store the mapping of terms to their postings
                    PositionalIndex: Each term keeps track of documents and positions of the terms occurring in the document

        Example:
            document1 = ['This', 'is' ,'a', 'dog', None]
            document2 = [None, 'This', 'is', 'a', 'cat']

            statistics = {
                'vocab'                     : Counter({'This': 2, 'is': 2, 'a' : 2, 'dog' : 1, 'cat': 1}),
                'unique_token_count'        : 5,
                'total_token_count'         : 10,
                'stored_total_token_count'  : 8,
                'number_of_documents'       : 2,
                'mean_document_length'      : 5
            }

            vocabulary = {'This', 'is', 'a', 'cat', 'dog'}

            document_metadata = {
                'document1': {'unique_tokens': 4, 'length': 5},
                'document2': {'unique_tokens': 4, 'length': 5}
            }

            If BasicInvertedIndex, we store 'term': [(docid, count), ...]
            index = {
                'This': [('document1', 1), ('document2', 1)],
                'is': [('document1', 1), ('document2', 1)],
                'a': [('document1', 1), ('document2', 1)],
                'dog': [('document1', 1)],
                'cat': [('document2', 1)]
            }
            If PositionalIndex, we store 'term': [(docid, count, [position]), ...]
            index = {
                'This': [('document1', 1, [0]), ('document2', 1, [1])],
                'is': [('document1', 1, [1]), ('document2', 1, [2])],
                'a': [('document1', 1, [2]), ('document2', 1, [3])],
                'dog': [('document1', 1, [3])],
                'cat': [('document2', 1, [4])]
            }

        """
        # Define necessary variables
        # We will use them later when we implement the below methods in BasicInvertedIndex and PositionalIndex classes
        self.statistics = {}
        self.statistics['vocab'] = Counter()
        self.statistics['total_token_count'] = 0
        self.statistics['stored_total_token_count'] = 0
        self.statistics['number_of_documents'] = 0
        self.statistics['unique_token_count'] = 0
        self.statistics['mean_document_length'] = 0
        self.vocabulary = set()
        self.document_metadata = {}
        self.index = defaultdict(list)

    # NOTE: The following functions have to be implemented in the two inherited classes and NOT in this class

    def add_doc(self, docid: int, tokens: list[str]) -> None:
        """
        Add a document to the index and update the index's metadata on the basis of this
        document's condition (e.g., collection size, average document length).

        Args:
            docid: The id of the document
            tokens: The tokens of the document
                Tokens that should not be indexed will have been replaced with None in this list.
                The length of the list should be equal to the number of tokens prior to any token removal.
        """
        # TODO: Implement this to add documents to the index
        raise NotImplementedError

    def remove_doc(self, docid: int) -> None:
        """
        Removes a document from the index and updates the index's metadata on the basis of this
        document's deletion.

        Args:
            docid: The id of the document
        """
        # TODO: Implement this to remove a document from the entire index and statistics
        raise NotImplementedError

    def get_postings(self, term: str) -> list:
        """
        Returns the list of postings, which contains (at least) all the documents that have that term.
        In most implementation, this information is represented as list of tuples where each tuple
        contains the docid and the term's frequency in that document.

        Args:
            term: The term to be searched for

        Returns:
            A list of tuples containing a document id for a document
            that had that search term and an int value indicating the term's frequency in
            the document
        """
        # TODO: Implement this to fetch a term's postings from the index
        raise NotImplementedError

    def get_doc_metadata(self, doc_id: int) -> dict[str, int]:
        """
        For the given document id, returns a dictionary with metadata about that document.
        Metadata should include keys such as the following:
            "unique_tokens": How many unique tokens are in the document (among those not-filtered)
            "length": how long the document is in terms of tokens (including those filtered)

        Args:
            docid: The id of the document

        Returns:
            A dictionary with metadata about the document
        """
        # TODO: Implement to fetch a particular document stored in metadata
        raise NotImplementedError

    def get_term_metadata(self, term: str) -> dict[str, int]:
        """
        For the given term, returns a dictionary with metadata about that term in the index.
        Metadata should include keys such as the following:
            "term_count": How many times this term appeared in the corpus as a whole
            "doc_frequency": How many documents contain this term

        Args:
            term: The term to be searched for

        Returns:
            A dictionary with metadata about the term in the index
        """
        # TODO: Implement to fetch a particular term stored in metadata
        raise NotImplementedError

    def get_statistics(self) -> dict[str, int]:
        """
        Returns a dictionary with properties and their values for the index.
        Keys should include AT LEAST the following:
            "unique_token_count": how many unique terms are in the index
            "total_token_count": how many total tokens are indexed including filterd tokens),
                i.e., the sum of the lengths of all documents
            "stored_total_token_count": how many total tokens are indexed excluding filterd tokens
            "number_of_documents": the number of documents indexed
            "mean_document_length": the mean number of tokens in a document (including filter tokens)

        Returns:
            A dictionary mapping statistical properties (named as strings) about the index to their values
        """
        # TODO: Calculate statistics like 'unique_token_count', 'total_token_count',
        #       'number_of_documents', 'mean_document_length' and any other relevant central statistic

        # Hint: Statistics only need recomputation if the index has been modified since the last calculation.
        #       This ensure substantial performance improvements.
        raise NotImplementedError

    def save(self, index_directory_name: str = 'tmp') -> None:
        """
        Saves the state of this index to the provided directory.
        The save state should include the inverted index as well as
        any metadata need to load this index back from disk.

        Args:
            index_directory_name: The name of the directory where the index will be saved
        """
        raise NotImplementedError

    def load(self, index_directory_name: str = 'tmp') -> None:
        """
        Loads the inverted index and any associated metadata from files located in the directory.
        This method will only be called after save() has been called, so the directory should
        match the filenames used in save(). Note that you call this function on an empty index object.

        Args:
            index_directory_name: The name of the directory that contains the index
        """
        raise NotImplementedError


class BasicInvertedIndex(InvertedIndex):
    def __init__(self) -> None:
        """
        This is the typical inverted index where each term keeps track of documents and the term count per document.
        This class will hold the mapping of terms to their postings.
        The class also has functions to save and load the index to/from disk and to access metadata about the index and the terms
        and documents in the index. These metadata will be necessary when computing your ranker functions.
        """
        super().__init__()
        self.statistics['index_type'] = 'BasicInvertedIndex'

    def add_doc(self, docid: int, tokens: list[str]) -> None:
        """
        Add a document to the index and update the index's metadata on the basis of this
        document's condition (e.g., collection size, average document length).

        Args:
            docid: The id of the document
            tokens: The tokens of the document
                Tokens that should not be indexed will have been replaced with None in this list.
                The length of the list should be equal to the number of tokens prior to any token removal.
        """
        document_counter = Counter(
            [token for token in tokens if token is not None])

        # Update the statistics
        self.statistics['vocab'].update(document_counter)
        self.statistics['unique_token_count'] = len(self.statistics['vocab'])
        self.statistics['total_token_count'] = self.statistics['total_token_count'] + \
            len(tokens)
        self.statistics['stored_total_token_count'] = self.statistics['stored_total_token_count'] + \
            document_counter.total()
        self.statistics['number_of_documents'] = self.statistics['number_of_documents'] + 1
        self.statistics['mean_document_length'] = self.statistics['total_token_count'] / \
            self.statistics['number_of_documents']

        # Update vocabulary
        self.vocabulary.update(document_counter.keys())

        # Update document metadata
        self.document_metadata[docid] = {
            'unique_tokens': len(document_counter),
            'length': len(tokens),
        }

        # Update index
        for token, count in document_counter.items():
            self.index[token].append((docid, count))

    def remove_doc(self, docid: int) -> None:
        """
        Removes a document from the index and updates the index's metadata on the basis of this
        document's deletion.

        Args:
            docid: The id of the document
        """
        if docid not in self.document_metadata:
            return

        document_counter = Counter()

        for token, count_list in self.index.items():
            for rec_docid, count in count_list:
                if docid == rec_docid:
                    document_counter[token] = count
                    break

        # Update the statistics
        self.statistics['vocab'] = self.statistics['vocab'] - document_counter
        self.statistics['unique_token_count'] = len(self.statistics['vocab'])
        self.statistics['total_token_count'] = self.statistics['total_token_count'] - \
            self.document_metadata[docid]['length']
        self.statistics['stored_total_token_count'] = self.statistics['stored_total_token_count'] - \
            document_counter.total()
        self.statistics['number_of_documents'] = self.statistics['number_of_documents'] - 1

        if self.statistics['number_of_documents'] > 0:
            self.statistics['mean_document_length'] = self.statistics['total_token_count'] / \
                self.statistics['number_of_documents']
        else:
            self.statistics['mean_document_length'] = 0

        # Update vocabulary
        self.vocabulary = set(self.statistics['vocab'])

        # Update document metadata
        del self.document_metadata[docid]

        # Update index
        for token, count in document_counter.items():
            self.index[token] = [(item, cnt)
                                 for item, cnt in self.index[token] if item != docid]

            if len(self.index[token]) == 0:
                del self.index[token]

    def get_postings(self, term: str) -> list:
        """
        Returns the list of postings, which contains (at least) all the documents that have that term.
        In most implementation, this information is represented as list of tuples where each tuple
        contains the docid and the term's frequency in that document.

        Args:
            term: The term to be searched for

        Returns:
            A list of tuples containing a document id for a document
            that had that search term and an int value indicating the term's frequency in
            the document
        """
        return self.index[term]

    def get_doc_metadata(self, doc_id: int) -> dict[str, int]:
        """
        For the given document id, returns a dictionary with metadata about that document.
        Metadata should include keys such as the following:
            "unique_tokens": How many unique tokens are in the document (among those not-filtered)
            "length": how long the document is in terms of tokens (including those filtered)

        Args:
            docid: The id of the document

        Returns:
            A dictionary with metadata about the document
        """
        return self.document_metadata[doc_id]

    def get_term_metadata(self, term: str) -> dict[str, int]:
        """
        For the given term, returns a dictionary with metadata about that term in the index.
        Metadata should include keys such as the following:
            "term_count": How many times this term appeared in the corpus as a whole
            "doc_frequency": How many documents contain this term

        Args:
            term: The term to be searched for

        Returns:
            A dictionary with metadata about the term in the index
        """
        return {
            "term_count": self.statistics["vocab"][term],
            "doc_frequency": len(self.index[term]),
        }

    def get_statistics(self) -> dict[str, int]:
        """
        Returns a dictionary with properties and their values for the index.
        Keys should include AT LEAST the following:
            "unique_token_count": how many unique terms are in the index
            "total_token_count": how many total tokens are indexed including filterd tokens),
                i.e., the sum of the lengths of all documents
            "stored_total_token_count": how many total tokens are indexed excluding filterd tokens
            "number_of_documents": the number of documents indexed
            "mean_document_length": the mean number of tokens in a document (including filter tokens)

        Returns:
            A dictionary mapping statistical properties (named as strings) about the index to their values
        """
        return self.statistics

    def save(self, index_directory_name: str = 'tmp') -> None:
        """
        Saves the state of this index to the provided directory.
        The save state should include the inverted index as well as
        any metadata need to load this index back from disk.

        Args:
            index_directory_name: The name of the directory where the index will be saved
        """
        # Create directory if it doesn't exist
        os.makedirs(index_directory_name, exist_ok=True)

        # Create data
        data = {
            "statistics": {
                **self.statistics,
                "vocab": dict(self.statistics["vocab"])
            },
            "vocabulary": list(self.vocabulary),
            "document_metadata": self.document_metadata,
            "index": dict(self.index)
        }

        file_path = os.path.join(index_directory_name, "index.json")

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def load(self, index_directory_name: str = 'tmp') -> None:
        """
        Loads the inverted index and any associated metadata from files located in the directory.
        This method will only be called after save() has been called, so the directory should
        match the filenames used in save(). Note that you call this function on an empty index object.

        Args:
            index_directory_name: The name of the directory that contains the index
        """
        file_path = os.path.join(index_directory_name, "index.json")

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.statistics = data["statistics"]
        self.statistics["vocab"] = Counter(self.statistics["vocab"])

        self.vocabulary = set(data["vocabulary"])

        # Document ID needs to be converted to int
        self.document_metadata = {
            int(docid): value
            for docid, value in data["document_metadata"].items()
        }

        # Restore index
        self.index = defaultdict(list)

        for term, count_list in data["index"].items():
            self.index[term] = [
                tuple(count_item) for count_item in count_list
            ]


class PositionalInvertedIndex(BasicInvertedIndex):
    def __init__(self) -> None:
        """
        This is the positional index where each term keeps track of documents and positions of the terms
        occurring in the document.
        """
        super().__init__()
        self.statistics['index_type'] = 'PositionalIndex'

    def add_doc(self, docid: int, tokens: list[str]) -> None:
        """
        Add a document to the index and update the index's metadata on the basis of this
        document's condition (e.g., collection size, average document length).

        Args:
            docid: The id of the document
            tokens: The tokens of the document
                Tokens that should not be indexed will have been replaced with None in this list.
                The length of the list should be equal to the number of tokens prior to any token removal.
        """
        positions_dict = defaultdict(list)

        for idx, token in enumerate(tokens):
            if token is not None:
                positions_dict[token].append(idx)

        document_counter = Counter({
            token: len(positions)
            for token, positions in positions_dict.items()
        })

        # Update the statistics
        self.statistics['vocab'].update(document_counter)
        self.statistics['unique_token_count'] = len(self.statistics['vocab'])
        self.statistics['total_token_count'] = self.statistics['total_token_count'] + \
            len(tokens)
        self.statistics['stored_total_token_count'] = self.statistics['stored_total_token_count'] + \
            document_counter.total()
        self.statistics['number_of_documents'] = self.statistics['number_of_documents'] + 1
        self.statistics['mean_document_length'] = self.statistics['total_token_count'] / \
            self.statistics['number_of_documents']

        # Update vocabulary
        self.vocabulary.update(document_counter.keys())

        # Update document metadata
        self.document_metadata[docid] = {
            'unique_tokens': len(document_counter),
            'length': len(tokens),
        }

        # Update index
        for token, positions in positions_dict.items():
            self.index[token].append((docid, len(positions), positions))

    def remove_doc(self, docid: int) -> None:
        """
        Removes a document from the index and updates the index's metadata on the basis of this
        document's deletion.

        Args:
            docid: The id of the document
        """
        if docid not in self.document_metadata:
            return

        document_counter = Counter()

        for token, count_list in self.index.items():
            for rec_docid, count, positions in count_list:
                if docid == rec_docid:
                    document_counter[token] = count
                    break

        # Update the statistics
        self.statistics['vocab'] = self.statistics['vocab'] - document_counter
        self.statistics['unique_token_count'] = len(self.statistics['vocab'])
        self.statistics['total_token_count'] = self.statistics['total_token_count'] - \
            self.document_metadata[docid]['length']
        self.statistics['stored_total_token_count'] = self.statistics['stored_total_token_count'] - \
            document_counter.total()
        self.statistics['number_of_documents'] = self.statistics['number_of_documents'] - 1

        if self.statistics['number_of_documents'] > 0:
            self.statistics['mean_document_length'] = self.statistics['total_token_count'] / \
                self.statistics['number_of_documents']
        else:
            self.statistics['mean_document_length'] = 0

        # Update vocabulary
        self.vocabulary = set(self.statistics['vocab'])

        # Update document metadata
        del self.document_metadata[docid]

        # Update index
        for token, count in document_counter.items():
            self.index[token] = [(item, cnt, positions)
                                 for item, cnt, positions in self.index[token] if item != docid]

            if len(self.index[token]) == 0:
                del self.index[token]

    def get_postings(self, term: str) -> list:
        """
        Returns the list of postings, which contains (at least) all the documents that have that term.
        In most implementation, this information is represented as list of tuples where each tuple
        contains the docid and the term's frequency in that document.

        Args:
            term: The term to be searched for

        Returns:
            A list of tuples containing a document id for a document
            that had that search term and an int value indicating the term's frequency in
            the document
        """
        return self.index[term]

    def get_doc_metadata(self, doc_id: int) -> dict[str, int]:
        """
        For the given document id, returns a dictionary with metadata about that document.
        Metadata should include keys such as the following:
            "unique_tokens": How many unique tokens are in the document (among those not-filtered)
            "length": how long the document is in terms of tokens (including those filtered)

        Args:
            docid: The id of the document

        Returns:
            A dictionary with metadata about the document
        """
        return self.document_metadata[doc_id]

    def get_term_metadata(self, term: str) -> dict[str, int]:
        """
        For the given term, returns a dictionary with metadata about that term in the index.
        Metadata should include keys such as the following:
            "term_count": How many times this term appeared in the corpus as a whole
            "doc_frequency": How many documents contain this term

        Args:
            term: The term to be searched for

        Returns:
            A dictionary with metadata about the term in the index
        """
        return {
            "term_count": self.statistics["vocab"][term],
            "doc_frequency": len(self.index[term]),
        }

    def get_statistics(self) -> dict[str, int]:
        """
        Returns a dictionary with properties and their values for the index.
        Keys should include AT LEAST the following:
            "unique_token_count": how many unique terms are in the index
            "total_token_count": how many total tokens are indexed including filterd tokens),
                i.e., the sum of the lengths of all documents
            "stored_total_token_count": how many total tokens are indexed excluding filterd tokens
            "number_of_documents": the number of documents indexed
            "mean_document_length": the mean number of tokens in a document (including filter tokens)

        Returns:
            A dictionary mapping statistical properties (named as strings) about the index to their values
        """
        return self.statistics

    def save(self, index_directory_name: str = 'tmp') -> None:
        """
        Saves the state of this index to the provided directory.
        The save state should include the inverted index as well as
        any metadata need to load this index back from disk.

        Args:
            index_directory_name: The name of the directory where the index will be saved
        """
        # Create directory if it doesn't exist
        os.makedirs(index_directory_name, exist_ok=True)

        # Create data
        data = {
            "statistics": {
                **self.statistics,
                "vocab": dict(self.statistics["vocab"])
            },
            "vocabulary": list(self.vocabulary),
            "document_metadata": self.document_metadata,
            "index": dict(self.index)
        }

        file_path = os.path.join(index_directory_name, "index.json")

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def load(self, index_directory_name: str = 'tmp') -> None:
        """
        Loads the inverted index and any associated metadata from files located in the directory.
        This method will only be called after save() has been called, so the directory should
        match the filenames used in save(). Note that you call this function on an empty index object.

        Args:
            index_directory_name: The name of the directory that contains the index
        """
        file_path = os.path.join(index_directory_name, "index.json")

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.statistics = data["statistics"]
        self.statistics["vocab"] = Counter(self.statistics["vocab"])

        self.vocabulary = set(data["vocabulary"])

        # Document ID needs to be converted to int
        self.document_metadata = {
            int(docid): value
            for docid, value in data["document_metadata"].items()
        }

        # Restore index
        self.index = defaultdict(list)

        for term, count_list in data["index"].items():
            self.index[term] = [
                tuple(count_item) for count_item in count_list
            ]


# NOTE (HW3): create_index() now has an OPTIONAL argument doc_augment_dict.
# This dict should be created from the `doc2query.csv` The keys are the document id and the values are the list of queries for a particular document. 
# The augmentation of the document should happen before all of the preprocessing steps, i.e., before stopwords removal or minimum word filtering. 
# Note that index-builders do not need to augment their documents so your code should support the cases where `doc_augment_dict` is or isn't provided.
class Indexer:
    '''
    The Indexer class is responsible for creating the index used by the search/ranking algorithm.
    '''
    @staticmethod
    def create_index(index_type: IndexType, dataset_path: str,
                     document_preprocessor: Tokenizer, stopwords: set[str],
                     minimum_word_frequency: int, text_key="text",
                     max_docs: int = -1, doc_augment_dict: dict[int, list[str]] | None = None) -> InvertedIndex:
        '''
        This function is responsible for going through the documents one by one and inserting them into the index after tokenizing the document

        Args:
            index_type: This parameter tells you which type of index to create, e.g., BasicInvertedIndex
            dataset_path: The file path to your dataset
            document_preprocessor: A class which has a 'tokenize' function which would read each document's text and return a list of valid tokens
            stopwords: The set of stopwords to remove during preprocessing or 'None' if no stopword filtering is to be done
            minimum_word_frequency: An optional configuration which sets the minimum word frequency of a particular token to be indexed
                If the token does not appear in the document at least for the set frequency, it will not be indexed.
                Setting a value of 0 will completely ignore the parameter.
            text_key: The key in the JSON to use for loading the text
            max_docs: The maximum number of documents to index
                Documents are processed in the order they are seen.
                Setting a value that is less than 1 (e.g. 0 or negative) will result in all documents being indexed.
            doc_augment_dict: An optional dictionary mapping docid to a list of strings to augment the document text with.
                If provided, the strings in the list will be prepended to the document TEXT (not text_key='title') before tokenization.
                If None, no augmentation will be done.

        Returns:
            An inverted index
    
        '''
        # TODO (HW3): This function now has an optional argument doc_augment_dict, add support for it.
        #             Please look at Readme.md for more details.
       
        # HINT: Think of what to do when doc_augment_dict exists, how can you deal with the extra information?
        #       How can you use that information with the tokens?
        #       If doc_augment_dict doesn't exist, it's the same as before, tokenizing just the document text

        # Choose indexer
        if index_type == IndexType.BasicInvertedIndex:
            index = BasicInvertedIndex()
        elif index_type == IndexType.PositionalIndex:
            index = PositionalInvertedIndex()
        else:
            raise ValueError("Index type not recognized.")

        # Use gzip if file is compressed
        if dataset_path.endswith(".jsonl.gz"):
            open_func = gzip.open
        else:
            open_func = open

        # Minimum count frequency implementation
        word_frequencies = Counter()

        if minimum_word_frequency > 0:
            with open_func(dataset_path, "rt", encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    if max_docs > 0 and idx >= max_docs:
                        break

                    document = json.loads(line)

                    # Document augmentation functionality
                    text = document[text_key]

                    if doc_augment_dict is not None and text_key != "title":
                        queries = doc_augment_dict.get(document["docid"], [])
                        if queries:
                            text = " ".join(queries) + " " + text

                    tokens = document_preprocessor.tokenize(text)
                    word_frequencies.update(tokens)

        # Stopword and minimum frequency filtering
        word_filter = set()

        if stopwords is not None:
            word_filter.update(stopwords)

        if minimum_word_frequency > 0:
            low_freq_words = {
                word
                for word, count in word_frequencies.items()
                if count < minimum_word_frequency
            }

            word_filter.update(low_freq_words)

        # Build index
        with open_func(dataset_path, "rt", encoding="utf-8") as f:

            for idx, line in tqdm(enumerate(f)):
                if max_docs > 0 and idx >= max_docs:
                    break

                document = json.loads(line)

                docid = document["docid"]
                text = document[text_key]

                # Document augmentation
                if doc_augment_dict is not None and text_key != "title":
                    queries = doc_augment_dict.get(document["docid"], [])
                    if queries:
                        text = " ".join(queries) + " " + text

                tokens = document_preprocessor.tokenize(text)

                filtered_tokens = [
                    None if token in word_filter else token
                    for token in tokens
                ]

                index.add_doc(docid, filtered_tokens)

        return index


'''
The following class is a stub class with none of the essential methods implemented. It is merely here as an example.
'''
class SampleIndex(InvertedIndex):
    '''
    This class does nothing of value
    '''

    def add_doc(self, docid, tokens):
        """Tokenize a document and add term ID """
        for token in tokens:
            if token not in self.index:
                self.index[token] = {docid: 1}
            else:
                self.index[token][docid] = 1

    def save(self):
        print('Index saved!')
