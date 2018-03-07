import gensim
from minisom import MiniSom
import numpy as np
import embedding_cluster
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))


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

sentences = ["I like to wear my tie to school everyday", "The game ended in a tie", "His kidnappers tied him to a chair","She tied a scarf around her neck","She tied knots in the rope","You need to tie your shoe","His hands and feet had been tied together","She tied the apron loosely around her waist","The team still has a chance to tie","I had the lead but he tied me by making a birdie on the last hole","Her time tied the world record","He tied the schools record in the high jump"]

def predict_group(sentences):
    predictions = []
    for sentence in sentences:
        sentence = sentence.lower()
        sentence_words = sentence.split(' ')
        sentence_words = [w for w in sentence_words if not w in stop_words]
        sentence_vectors = np.array([model.get_vector(w) for w in sentence_words])
        average_vector = np.mean(sentence_vectors, axis = 0)
        predictions.append(kmeans.predict([average_vector]))
        
    return predictions

predictions = predict_group(sentences)






