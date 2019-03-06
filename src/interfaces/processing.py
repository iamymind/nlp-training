from abc import ABC, abstractmethod


class ProcessingPipeline(ABC):
    """
    ProcessingPipeline abstract class provides interface for building data processing pipelines
    """

    @abstractmethod
    def run_pipeline(self):
        """
        Abstract method returning processed text data in required form
        :return: processed text data
        """
        pass
