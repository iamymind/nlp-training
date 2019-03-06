from src.interfaces.data import TextReader
from src.interfaces.processing import ProcessingPipeline


class PlainStringTextReader(TextReader):

    def __init__(self, path: str, pipeline: ProcessingPipeline):
        """
        Simple single string text reader class. Reads files and processes the text in accordance with injected
        pipeline object
        :param path: Text data file path string
        :param pipeline: Instance of the ProcessingPipeline abstract class, to be substituted with concrete
         pipeline class instance
        """
        self._path = path
        self._pipeline = pipeline

    def get_text(self) -> str:
        """
        Opens plain text string data file, reads text, closes file
        :return: Text string from file
        """
        with open(self.path, 'r') as text_file:
            raw_text = text_file.read()
            return self.pipeline.run_pipeline(raw_text)

    @property
    def path(self) -> str:
        """
        Getter for data path string object
        :return: path string
        """
        return self._path

    @path.setter
    def path(self, path: str) -> None:
        """
        Sets path string variable
        :param path: path string
        :return: None
        """
        self._path = path

    @property
    def pipeline(self) ->ProcessingPipeline:
        """Getter for pipeline object"""
        return self._pipeline

    @pipeline.setter
    def pipeline(self, pipeline: ProcessingPipeline) -> None:
        self._pipeline = pipeline
