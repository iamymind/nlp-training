from src.processing_pipelines import BasicStringTextProcessingPipeline
from src.readers import PlainStringTextReader
from src.skipgram_data_preprocessing import SkipGramBatcher
from src.algorithms import Word2Vec

# An example of entire word vectors training process on sample text
pipeline = BasicStringTextProcessingPipeline()
reader = PlainStringTextReader('text.txt', pipeline)
text = reader.get_text()
batcher = SkipGramBatcher(text)
w2v = Word2Vec(batcher)

w2v.n_samples = 300
w2v.learning_rate = 0.0001
w2v.n_epochs = 20
w2v.n_embeddings = 500

w2v.train()
w2v.show_vectors(n_words=300, alpha=0.3)
