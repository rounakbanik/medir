"""
This is the template for implementing the tokenizer for your search engine.
You will be testing some tokenization techniques.
"""
from nltk.tokenize import RegexpTokenizer
import spacy
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration


class Tokenizer:
    def __init__(self, lowercase: bool = True, multiword_expressions: list[str] = None) -> None:
        """
        A generic class for objects that turn strings into sequences of tokens.
        A tokenizer can support different preprocessing options or use different methods
        for determining word breaks.

        Args:
            lowercase: Whether to lowercase all the tokens
            multiword_expressions: A list of strings that should be recognized as single tokens
                If set to 'None' no multi-word expression matching is performed.
        """
        # TODO: Save arguments that are needed as fields of this class
        self.lowercase = lowercase
        self.multiword_expressions = multiword_expressions

    def postprocess(self, input_tokens: list[str]) -> list[str]:
        """
        Performs any set of optional operations to modify the tokenized list of words such as
        lower-casing and multi-word-expression handling. After that, return the modified list of tokens.

        Args:
            input_tokens: A list of tokens

        Returns:
            A list of tokens processed by lower-casing depending on the given condition

        Examples:
            If lowercase, "Taylor" "Swift" -> "taylor" "swift"
            If "Taylor Swift" in multiword_expressions, "Taylor" "Swift" -> "Taylor Swift"
        """
        # TODO: Add support for lower-casing and multi-word expressions
        if self.multiword_expressions:
            multiword_expressions = self.multiword_expressions
        else:
            multiword_expressions = []

        if self.lowercase:
            input_tokens = [
                x.lower() for x in input_tokens
            ]
            multiword_expressions = [
                x.lower() for x in multiword_expressions
            ]

        multiword_expressions = [
            x.split() for x in multiword_expressions
        ]

        if len(multiword_expressions) == 0:
            return input_tokens

        mwe = MWETokenizer(mwes=multiword_expressions, separator=" ")

        processed_tokens = mwe.tokenize(input_tokens)

        return processed_tokens

    def tokenize(self, text: str) -> list[str]:
        """
        Splits a string into a list of tokens and performs all required postprocessing steps.

        Args:
            text: An input text you want to tokenize

        Returns:
            A list of tokens
        """
        # NOTE: You should implement this in a subclass, not here
        raise NotImplementedError(
            'tokenize() is not implemented in the base class; please use a subclass')


class SampleTokenizer(Tokenizer):
    def tokenize(self, text: str) -> list[str]:
        """This is a dummy tokenizer.

        Args:
            text: An input text you want to tokenize

        Returns:
            A list of tokens
        """
        return ['token_1', 'token_2', 'token_3']  # This is not correct; it is just a placeholder.


class SplitTokenizer(Tokenizer):
    def __init__(self, lowercase: bool = True, multiword_expressions: list[str] = None) -> None:
        """
        Uses the split function to tokenize a given string.

        Args:
            lowercase: Whether to lowercase all the tokens
            multiword_expressions: A list of strings that should be recognized as single tokens
                If set to 'None' no multi-word expression matching is performed.
                No need to perform/implement multi-word expression recognition for HW3; you can ignore this.
        """
        super().__init__(lowercase, multiword_expressions)

    def tokenize(self, text: str) -> list[str]:
        """
        Split a string into a list of tokens using whitespace as a delimiter.

        Args:
            text: An input text you want to tokenize

        Returns:
            A list of tokens

        Examples:
            If lowercase = False, multiword_expressions = None, "This is an apple" -> "This" "is" "an" "apple"
        """
        return self.postprocess(text.split())


class RegexTokenizer(Tokenizer):
    def __init__(self, token_regex: str = r'\w+', lowercase: bool = True, multiword_expressions: list[str] = None) -> None:
        """
        The Natural Language Toolkit (NLTK) is a Python package for natural language processing.
        To learn more, visit https://pypi.org/project/nltk/

        Installation Instructions:
            It is recommended to install packages in a virtual environment.
            Here is an example to do so:
                $ python -m venv [your python virtual enviroment]
                $ source [your python virtual enviroment]/bin/activate # or [your python virtual environment]\Scripts\activate on Windows
                $ pip install -U nltk

        Your tasks:
        Uses NLTK's RegexpTokenizer to tokenize a given string.

        Args:
            token_regex: Use the following default regular expression pattern: '\w+'
            lowercase: Whether to lowercase all the tokens
            multiword_expressions: A list of strings that should be recognized as single tokens
                If set to 'None' no multi-word expression matching is performed.
                No need to perform/implement multi-word expression recognition for HW3; you can ignore this.
        """
        super().__init__(lowercase, multiword_expressions)
        # TODO: Save a new argument that is needed as a field of this class
        self.token_regex = token_regex

        # TODO: Initialize the NLTK's RegexpTokenizer
        self.tokenizer = RegexpTokenizer(self.token_regex)

    def tokenize(self, text: str) -> list[str]:
        """
        Uses NLTK's RegexTokenizer and a regular expression pattern to tokenize a string.

        Args:
            text: An input text you want to tokenize

        Returns:
            A list of tokens
        """
        # TODO: Tokenize the given text and perform postprocessing on the list of tokens
        #       using the postprocess function
        tokens = self.tokenizer.tokenize(text)
        tokens = self.postprocess(tokens)

        return tokens


class SpaCyTokenizer(Tokenizer):
    def __init__(self, lowercase: bool = True, multiword_expressions: list[str] = None) -> None:
        """
        spaCy is a free, open-source library for advanced Natural Language Processing (NLP) in Python.
        To learn more, visit: https://spacy.io/

        Installation Instructions:
            Please visit https://spacy.io/usage
            It is recommended to install packages in a virtual environment.
            Here is an example to do so:
                $ python -m venv [your python virtual enviroment]
                $ source [your python virtual enviroment]/bin/activate # or [your python virtual environment]\Scripts\activate on Windows
                $ pip install -U pip setuptools wheel
                $ pip install -U spacy

        After installation you typically want to download a trained pipeline 
        (e.g. "en_core_web_sm", which is an English pipeline)
        To learn more, visit https://spacy.io/models/en
            $ python -m spacy download en_core_web_sm

        To use "en_core_web_sm"
            >>> import spacy
            >>> nlp = spacy.load("en_core_web_sm")

        Your tasks: 
        Use a spaCy tokenizer to convert named entities into single words. 
        Check the spaCy documentation to learn about the feature that supports named entity recognition.
        Hint: You may want to take a look at merge_entities function in https://spacy.io/api/pipeline-functions

        Args:
            lowercase: Whether to lowercase all the tokens
            multiword_expressions: A list of strings that should be recognized as single tokens
                If set to 'None' no multi-word expression matching is performed.
                No need to perform/implement multi-word expression recognition for HW3; you can ignore this.
        """
        super().__init__(lowercase, multiword_expressions)
        self.nlp = spacy.load("en_core_web_sm")
        self.nlp.add_pipe("merge_entities")

    def tokenize(self, text: str) -> list[str]:
        """
        Use a spaCy tokenizer to convert named entities into single words.

        Args:
            text: An input text you want to tokenize

        Returns:
            A list of tokens
        """
        tokens = [t.text for t in self.nlp(text)]
        tokens = self.postprocess(tokens)
        return tokens


# DONE (HW3): Take in a doc2query model and generate queries from a piece of text
# NOTE: This is just to check you can use the models;
#       for downstream tasks such as index augmentation with the queries, use doc2query.csv
class Doc2QueryAugmenter:
    """
    Requirements:
        Please install pytorch to your virtual environment (Assuming you are on Great Lakes using Linux).
            $ pip3 install torch torchvision
        
        Verification whether pytorch is installed correctly
            >>> import torch
            >>> x = torch.rand(5, 3)
            >>> print(x)

        The output should be something similar to:
            tensor([[0.3380, 0.3845, 0.3217],
                    [0.8337, 0.9050, 0.2650],
                    [0.2979, 0.7141, 0.9069],
                    [0.1449, 0.1132, 0.1375],
                    [0.4675, 0.3947, 0.1426]])

        Visit https://pytorch.org/get-started/locally/ for other operating systems.

        Please install transformers to your virtual environment.
            $ pip3 install "transformers[torch]"
        
        Visit https://pypi.org/project/transformers/ for more details.

    Overview:    
        This class is responsible for generating queries for a document.
        These queries can augment the document before indexing.

        MUST READ: https://huggingface.co/doc2query/msmarco-t5-base-v1

        OPTIONAL reading
            1. Document Expansion by Query Prediction (Nogueira et al.): https://arxiv.org/pdf/1904.08375.pdf
    """
    
    def __init__(self, doc2query_model_name: str = 'doc2query/msmarco-t5-base-v1') -> None:
        """
        Creates the T5 model object and the corresponding dense tokenizer.
        
        Args:
            doc2query_model_name: The name of the T5 model architecture used for generating queries
        """
        self.device = torch.device('cpu')  # Do not change this unless you know what you are doing

        # DONE (HW3): Create the dense tokenizer and query generation model using HuggingFace transformers
        # NOTE: See https://huggingface.co/doc2query/msmarco-t5-base-v1 for details
        
        self.tokenizer = T5Tokenizer.from_pretrained(doc2query_model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(doc2query_model_name)


    def get_queries(self, document: str, n_queries: int = 5, prefix_prompt: str = '') -> list[str]:
        """
        Overview:
            For the given model, generate a list of queries that might reasonably be issued to search for that document
                using the pre-trained T5 model.

        Steps
            1. Use the dense tokenizer/encoder to create the dense document vector.
            2. Use the T5 model to generate the dense query vectors (you should have a list of vectors).
            3. Decode the query vector using the tokenizer/decode to get the appropriate queries.
            4. Return the queries.
         
            Ensure you take care of edge cases.
         
        OPTIONAL (DO NOT DO THIS before you finish the assignment):
            Neural models are best performing when batched to the GPU.
            Try writing a separate function which can deal with batches of documents.
            DO NOT submit the GPU code to the autograder.
        
        Args:
            document: The text from which queries are to be generated
            n_queries: The total number of queries to be generated
            prefix_prompt: An optional parameter that gets added before the text.
                Some models like flan-t5 are not fine-tuned to generate queries.
                So we need to add a prompt to instruct the model to generate queries.
                This string enables us to create a prefixed prompt to generate queries for the models.
                **See the PDF of the HW3 spec for what you need to do for this part.**
                Prompt-engineering: https://en.wikipedia.org/wiki/Prompt_engineering
        
        Returns:
            A list of query strings generated from the text
        """
        # Define default parameters
        # NOTE: Feel free to change these values to experiment
        document_max_token_length = 400  # as used in OPTIONAL Reading 1 (passed as max_length to the model's generate function)
        top_p = 0.85

        # NOTE: See https://huggingface.co/doc2query/msmarco-t5-base-v1 for details

        # DONE (HW3): Return an empty list if 0 queries are requested to be generated or if the document is empty
        if document == "" or n_queries <= 0:
            return []
        

        # DONE (HW3): Use the dense tokenizer/encoder to create the dense document vector.
        #     NOTE: Leave max_length unset, this will use the predefined model maximum length
        #     NOTE: Set truncation=True to truncate any text longer than the model's maximum length
        #     NOTE: Set return_tensors='pt' to get PyTorch tensors
        text = prefix_prompt + document
        input_ids = self.tokenizer.encode(
            text, 
            truncation=True, 
            return_tensors='pt')


        # DONE (HW3): Use the T5 model to generate the dense query vectors (you should get a list of vectors).
        #     NOTE: Set max_length to document_max_token_length
        #     NOTE: Set do_sample=True to use sampling
        #     NOTE: Figure out how to use top_p and n_queries
        outputs = self.model.generate(
            input_ids=input_ids,
            max_length=document_max_token_length,
            do_sample=True,
            top_p=top_p,
            num_return_sequences=n_queries,
        )


        # DONE (HW3): Decode the query vector using the tokenizer/decode to get the appropriate queries.
        #     NOTE: Set skip_special_tokens=True to remove any special tokens
        queries = []
        for i in range(len(outputs)):
            query = self.tokenizer.decode(outputs[i], skip_special_tokens=True)
            queries.append(query)


        # DONE (HW3): Return the queries.
        return queries



# Don't forget that you can have a main function here to test anything in the file
if __name__ == '__main__':
    pass
