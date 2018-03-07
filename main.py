import gensim
from minisom import MiniSom
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans,AffinityPropagation
import numpy as np

model = gensim.models.KeyedVectors.load_word2vec_format('glove.twitter.27B.200d.Word2Vecformat.txt')


test_word = 'tie'

nearest_words = model.most_similar(positive = [test_word], topn = 1000)
nearest_vectors = []

for i in nearest_words:
    vector = model.get_vector(i[0])
    nearest_vectors.append(vector)
   
"""
som = MiniSom(10, 10, 200, sigma=0.3, learning_rate=0.5) # initialization of 6x6 SOM
print("Training...")
som.train_random(nearest_vectors, 100) # trains the SOM with 100 iterations
print("...ready!")
"""


nearest_vectors = np.array(nearest_vectors)

kmeans = KMeans(n_clusters = 5, random_state = 0).fit(nearest_vectors)
results = kmeans.labels_

affinity = AffinityPropagation()
affinity.fit(nearest_vectors)
results_affinity = affinity.labels_


groups = [[],[],[],[],[]]
groups_affinity = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]

for i in range(len(results)):
    groups[results[i]].append(nearest_words[i])
    
for i in range(len(results_affinity)):
    print(i)
    groups_affinity[results_affinity[i]].append(nearest_words[i])
    print(results_affinity[i])
    print(groups_affinity)

pca = PCA(n_components = 2)
pca.fit(nearest_vectors)
reduced = pca.transform(nearest_vectors)

xs = reduced[:, 0]
ys = reduced[:, 1]

# draw
colors = ['r','g','b','y','m']
plt.figure(figsize=(12,8))
plt.scatter(xs, ys, marker = 'o')
for i, w in enumerate(nearest_words):
    plt.annotate(
        w,
        xy = (xs[i], ys[i]), xytext = (3, 3),
        textcoords = 'offset points', ha = 'left', va = 'top', color = colors[results[i]])

plt.show()
