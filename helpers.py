import re
from pipe import Pipe
from nltk.corpus import stopwords


@Pipe
def lower_case(text: str) -> str:
    """
    Pipeline element function. Low cases passed text
    :param text: Text string
    :return: Text string
    """
    return text.lower().strip(' ')


@Pipe
def remove_special_chars(text: str) -> str:
    """
    Pipeline element function. Removes special characters from passed text
    :param text: Text string
    :return: Text string
    """
    return re.sub(r'[^\w\s]', '', text)


@Pipe
def tokenize(text: str) -> list:
    """
    Pipeline element function. Tokenizes passed text
    :param text: Text string
    :return: List of tokens
    """
    text = re.sub("\s\s+", " ", text)
    return text.split(' ')


@Pipe
def remove_stopwords(text: list, lang='english', stop_words=stopwords) -> list:
    """
    Pipeline element function. Removes stop words from text
    :param text: List of text tokenz
    :param lang: Language of stop words. Default: English
    :param stop_words: Data loader nltk library class object
    :return: List of tokens
    """
    stop_words = stop_words.words(lang)
    return [word for word in text if word not in stop_words]
