from typing import Tuple

import numpy as np
from src.interfaces.data import Batcher
from collections import Counter
import random


class SkipGramBatcher(Batcher):

    def __init__(self, text: list, freq_threshold: float=1e-3, window_size: int=3, batch_size: int=3):
        """
        Implementation of batcher class for skip gram word2vec algorithm. The batcher generates chunks of center and
        context words for a given text corpus

        :param text: List of text corpus tokens
        :param freq_threshold: float number to specify frequent words filtering threshold as specified in the paper:
        https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf
        :param window_size: Integer value of window of context words around center word
        :param batch_size: Integer value of the batch: number of pairs od center and context words in a batch
        """
        self._text_corpus = text
        self._freq_threshold = freq_threshold
        self._window_size = window_size
        self._batch_size = batch_size
        self._numericalized_text, \
        self._int_key_vocabulary, \
        self._word_key_vocabulary = self.__numericalize_text(self._text_corpus)
        self._freq_filtered_corpus = self.__words_frequency_filter()

    def __numericalize_text(self, text: list) -> Tuple:
        """
        Converts list of text tokens into list of word numeric unique ids. ids are equal to word count of a certain
        word in corpus

        :param text: List of text tokens
        :return: Tuple of list of numeric word tokens, dictionary key:word id, value:word text,
        dictionary key:word text, value: word id
        """
        word_counts = Counter(text)
        vocabulary_sorted = sorted(word_counts, key=word_counts.get, reverse=True)
        int_key_vocabulary = {idx: word for idx, word in enumerate(vocabulary_sorted)}
        word_key_vocabulary = {word: idx for idx, word in int_key_vocabulary.items()}
        numericalized_text = [word_key_vocabulary[word] for word in text]
        return numericalized_text, int_key_vocabulary, word_key_vocabulary

    def __words_frequency_filter(self) -> list:
        """
        Filters the most frequent words in corpus to prevent them appearing in batches

        :return: List of frequency filtered numeric ids of text tokens
        """
        word_counts = Counter(self._numericalized_text)
        total_count = len(self._numericalized_text)
        freqs = {word: count / total_count for word, count in word_counts.items()}
        p_drop = {word: 1 - np.sqrt(self._freq_threshold / freqs[word]) for word in word_counts}
        freq_filtered_corpus = [word for word in self._numericalized_text if random.random() < (1 - p_drop[word])]
        return freq_filtered_corpus

    def __get_center_word_neighbours_num(self) -> int:
        """
        Generates random number with near normal distribution to randomly identify number of
        neighbouring context words around a given center word

        :return: Randomly generated number of neighbour context words around central word
        """
        num_neighbours = abs(int(np.random.normal(1, self.window_size/3)))
        if (num_neighbours > self.window_size) | (num_neighbours < 1):
            num_neighbours = random.randint(1, self.window_size)
        return num_neighbours

    def __get_target_contexts(self, idx: int) -> list:
        """
        Method finds context words surrounding a center word which id is passed to the method

        :param idx: Integer index number of center word in word corpus tokens list
        :return: List of numeric ids of context words surrounding a given central word
        """
        neighbourhood_radius = self.__get_center_word_neighbours_num()
        start = idx - neighbourhood_radius if (idx - neighbourhood_radius) > 0 else 0
        stop = idx + neighbourhood_radius
        context_words = self._freq_filtered_corpus[start:idx] + self._freq_filtered_corpus[idx+1:stop+1]
        return context_words

    def get_batch(self) -> Tuple:
        """
        Generates batches of 2 lists. List inputs contains a batch of center words. List labels contains
        corresponding context word for every center word in the input list

        :return: Tuple: list of center words, list of context words for each center word in batch
        """
        n_batches = len(self.freq_filtered_corpus)//self.batch_size
        words = self.freq_filtered_corpus[:n_batches*self.batch_size]
        for idx in range(0, len(words), self.batch_size):
            batch = words[idx:idx + self.batch_size]
            inputs, labels = [], []
            for i in range(len(batch)):
                word_global_index = idx + i
                batch_inputs = batch[i]
                batch_labels = self.__get_target_contexts(word_global_index)
                inputs.extend([batch_inputs]*len(batch_labels))
                labels.extend(batch_labels)
            yield inputs, labels

    @property
    def text_corpus(self) -> list:
        return self._text_corpus

    @property
    def numericalized_text(self) -> list:
        return self._numericalized_text

    @property
    def int_key_vocabulary(self) -> dict:
        return self._int_key_vocabulary

    @property
    def word_key_vocabulary(self) -> dict:
        return self._word_key_vocabulary

    @property
    def freq_filtered_corpus(self) -> list:
        return self._freq_filtered_corpus

    @property
    def freq_threshold(self) -> float:
        return self._freq_threshold

    @property
    def window_size(self) -> float:
        return self._window_size

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @text_corpus.setter
    def text_corpus(self, text: list) -> None:
        self._text_corpus = text
        self._numericalized_text, \
        self._int_key_vocabulary, \
        self._word_key_vocabulary = self.__numericalize_text(self._text_corpus)
        self._freq_filtered_corpus = self.__words_frequency_filter()

    @freq_threshold.setter
    def freq_threshold(self, freq_threshold: float) -> None:
        self._freq_threshold = freq_threshold
        self._freq_filtered_corpus = self.__words_frequency_filter()

    @window_size.setter
    def window_size(self, window_size: int) -> None:
        self._window_size = window_size

    @batch_size.setter
    def batch_size(self, batch_size: int) -> None:
        self._batch_size = batch_size