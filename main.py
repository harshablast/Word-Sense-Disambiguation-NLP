import gensim
from minisom import MiniSom
import numpy as np
import embedding_cluster

model = gensim.models.KeyedVectors.load_word2vec_format('glove.twitter.27B.200d.Word2Vecformat.txt')


test_word = 'tie'

nearest_words = model.most_similar(positive = [test_word], topn = 1000)
nearest_vectors = []

for i in nearest_words:
    vector = model.get_vector(i[0])
    nearest_vectors.append(vector)

nearest_vectors = np.array(nearest_vectors)


"""
som = MiniSom(10, 10, 200, sigma=0.3, learning_rate=0.5) # initialization of 6x6 SOM
print("Training...")
som.train_random(nearest_vectors, 100) # trains the SOM with 100 iterations
print("...ready!")
"""

groups, results = embedding_cluster.get_clusters(nearest_vectors, nearest_words)
embedding_cluster.plot_clusters(nearest_vectors, nearest_words, results)

sentence1 = "I like to wear my tie to school everyday"
sentence2 = "The game ended in a tie"








