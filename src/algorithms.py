from typing import Tuple

from src.interfaces.data import Batcher
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


class Word2Vec:

    def __init__(self, batcher: Batcher, lr:float=1e-2, n_embeddings:int=300, n_samples:int=500, n_epochs:int=5):
        """
        Implementation of Word2Vec algorithm with negative sampling

        :param batcher: Batcher object
        :param lr: Learning rate
        :param n_embeddings: Dimensionality of embeddings
        :param n_samples: Number of samples to be used in negative sampling
        :param n_epochs: Number of train epochs
        """
        self._batcher = batcher
        self._lr = lr
        self._n_embeddings = n_embeddings
        self._n_samples = n_samples
        self._n_epochs = n_epochs
        self._graph,\
        self._inputs,\
        self._labels,\
        self._embeddings_matrix,\
        self._embedings_vectors,\
        self._context_matrix,\
        self._context_bias,\
        self._loss,\
        self._cost,\
        self._optimizer,\
        self._optimization_step = self.__build_graph()
        self._train_status = False
        self._trained_embeddings = None

    def __build_graph(self) -> Tuple:
        """
        Builds computational graph for Word2Vec network

        :return: Tuple with declared elements of the computational graph
        """
        graph = tf.Graph()
        vocab_size = len(self.batcher.int_key_vocabulary)
        with graph.as_default():
            inputs = tf.placeholder(tf.int32, [None], name='inputs')
            labels = tf.placeholder(tf.int32, [None, None], name='labels')
            embeddings_matrix = tf.Variable(tf.random_uniform((vocab_size, self._n_embeddings), -1, 1))
            embedings_vectors = tf.nn.embedding_lookup(embeddings_matrix, inputs)
            context_matrix = tf.Variable(tf.random_uniform((vocab_size, self._n_embeddings)),
                                         name="context_matrix_weights")
            context_bias = tf.Variable(tf.zeros(vocab_size), name="context_matrix_biases")
            loss = tf.nn.sampled_softmax_loss(weights=context_matrix ,
                                              biases=context_bias,
                                              labels=labels,
                                              inputs=embedings_vectors,
                                              num_sampled=self._n_samples,
                                              num_classes=vocab_size)
            cost = tf.reduce_mean(loss)
            optimizer = tf.train.AdamOptimizer(learning_rate=self._lr)
            optimization_step = optimizer.minimize(cost)
        return graph,\
               inputs,\
               labels,\
               embeddings_matrix,\
               embedings_vectors,\
               context_matrix,\
               context_bias,\
               loss,\
               cost, \
               optimizer,\
               optimization_step

    def train(self) -> None:
        """
        Trains Word2Vec network. Updates embeddings matrix after training

        :return: None
        """
        with tf.Session(graph=self._graph) as sess:
            sess.run(tf.global_variables_initializer())
            iteration = 0
            for epoch in range(1, self._n_epochs + 1):
                epoch_loss_sum = 0
                epoch_avg_loss = 0
                batch = self._batcher.get_batch()
                for center_words, context_words in batch:
                    iteration += 1
                    feed = {self._inputs: center_words, self._labels: np.array(context_words)[:, None]}
                    train_loss, _ = sess.run([self._cost, self._optimization_step], feed_dict=feed)
                    epoch_loss_sum += train_loss
                    epoch_avg_loss = epoch_loss_sum/iteration
                    if iteration % 1000 == 0:
                        print("Iteration: {0}; Avg Epoch Loss: {1}".format(iteration, epoch_avg_loss))
                if epoch % 1 == 0:
                    print("Epoch: {0}; Avg Epoch Loss: {1}".format(epoch, epoch_avg_loss))
            self._trained_embeddings = sess.run(self._embeddings_matrix)
            self._train_status = True

    def show_vectors(self, n_words:int=100, figsize: Tuple=(15, 15), color: str='steelblue', alpha: float=0.7) -> None:
        """
        Draws 2-D representation of word eembeddings of the selected number of words in the dictionary

        :param n_words: Number of words to be shown on the scatterplot
        :param figsize: Width and height of the scatter plot figure
        :param color: Color of scatters
        :param alpha: Transparency of scatters
        :return: None
        """
        show_words = n_words
        tsne = TSNE()
        embeddings_tsne = tsne.fit_transform(self._trained_embeddings[:show_words, :])
        fig, ax = plt.subplots(figsize=figsize)
        for idx in range(show_words):
            plt.scatter(*embeddings_tsne[idx, :], color=color)
            plt.annotate(self._batcher.int_key_vocabulary[idx],
                         (embeddings_tsne[idx, 0],
                          embeddings_tsne[idx, 1]),
                         alpha=alpha)
        plt.show()

    @property
    def batcher(self) -> Batcher:
        return self._batcher

    @property
    def embeddings(self) -> np.array:
        if self._train_status:
            return self._trained_embeddings
        else:
            print("Embeddings have not been trained yet")

    @property
    def n_embeddings(self) -> int:
        return self._n_embeddings

    @property
    def learning_rate(self) -> int:
        return self._lr

    @property
    def n_samples(self) -> int:
        return self._n_samples

    @property
    def n_epochs(self) -> int:
        return self._n_epochs

    @n_epochs.setter
    def n_epochs(self, n_epochs: int) -> None:
        self._n_epochs = n_epochs

    @n_samples.setter
    def n_samples(self, n_samples: int) -> None:
        self._n_samples = n_samples

    @learning_rate.setter
    def learning_rate(self, learning_rate: int) -> None:
        self._lr = learning_rate

    @batcher.setter
    def batcher(self, batcher: Batcher) -> None:
        self._batcher = batcher

    @n_embeddings.setter
    def n_embeddings(self, n_embeddings: int) -> None:
        self._n_embeddings = n_embeddings



