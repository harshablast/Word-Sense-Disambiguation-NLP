from sklearn.cluster import KMeans,AffinityPropagation
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def get_clusters(nearest_vectors, nearest_words):

    kmeans = KMeans(n_clusters = 5, random_state = 0).fit(nearest_vectors)
    results = kmeans.labels_

    #affinity = AffinityPropagation()
    #affinity.fit(nearest_vectors)
    #results_affinity = affinity.labels_


    groups = [[],[],[],[],[]]
    #groups_affinity = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
    
    for i in range(len(results)):
        groups[results[i]].append(nearest_words[i])
    """
    for i in range(len(results_affinity)):
        groups_affinity[results_affinity[i]].append(nearest_words[i])
    """
    
    return groups, results

def plot_clusters(nearest_vectors, nearest_words, results):
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