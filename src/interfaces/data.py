from abc import ABC, abstractmethod


class TextReader(ABC):
    """
    TextReader abstract class provides interface for text files reading classes
    """

    @abstractmethod
    def get_text(self):
        """
        Abstract method returning preprocessed text in required form
        :return: text data in required types depending on requirements
        """
        pass


class Batcher(ABC):
    """
    Batcher abstract class provides interface data batching classes
    """
    @abstractmethod
    def get_batch(self):
        pass

    @abstractmethod
    def int_key_vocabulary(self):
        pass