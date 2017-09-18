import gensim

# gensim: This package contains interfaces and functionality to computer pair-wise document similarities within
# a corpus of documents.
# KeyedVectors: Class to contain vectors vocab for the Word2Vec training class and other w2v methods
# not involved in training such as most_similar()
# load_word2vec_format: Load the input-hidden wight matrix from the original C word2vec-tool format.
w2v = gensim.models.KeyedVectors.load_word2vec_format('./vectors.txt', binary=False)
# Find the top-N most similar words. Positive words contribute positively towards the similarity,
# negative words negatively.
# This method computes cosine similarity between a simple mean of the projection weight vectors of the given words
# and the vectors for word-analogy and distance scripts in the original word2vec implementation.
for vectors in w2v.most_similar(positive=['mac']):
    print(vectors)
