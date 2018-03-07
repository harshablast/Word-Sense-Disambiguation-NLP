import gensim
from gensim.scripts.glove2word2vec import glove2word2vec

glove2word2vec('glove.twitter.27B.200d.txt', 'glove.twitter.27B.200d.Word2Vecformat.txt')