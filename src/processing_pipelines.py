from src.interfaces.processing import ProcessingPipeline
from helpers import lower_case, remove_special_chars, tokenize, remove_stopwords


class BasicStringTextProcessingPipeline(ProcessingPipeline):

    def run_pipeline(self, text: str) -> list:
        """
        Implements abstract method: creates standard text processing pipeline from text processing helper functions
        :param text: Text string data
        :return: Preprocessed text tokens list
        """
        text_tokens = text | lower_case | remove_special_chars | tokenize | remove_stopwords
        return text_tokens
