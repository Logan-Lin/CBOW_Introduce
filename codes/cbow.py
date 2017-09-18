import gensim
import keras.backend as K
import numpy as np
from keras.layers import Dense, Embedding, Lambda
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils, plot_model

np.random.seed(13)

# path = get_file('alice.txt', origin='http://www.gutenberg.org/cache/epub/11/pg11.txt')
path = "test.txt"
corpus = open(path).readlines()[0:200]
corpus = [sentence for sentence in corpus if sentence.count(' ') >= 2]

# Tokenizer: This class allows to vectorize a text corpus, by turning each text into either a sequence of integers
# or into a vector.
tokenizer = Tokenizer()
# Updates internal vocabulary based on a list of texts.
tokenizer.fit_on_texts(corpus)
# Transforms each text in texts in a sequence of integers.
corpus = tokenizer.texts_to_sequences(corpus)
nb_samples = sum(len(s) for s in corpus)
V = len(tokenizer.word_index) + 1
dim = 100
window_size = 2


def generate_data(corpus, window_size, V):
    maxlen = window_size * 2
    for words in corpus:
        contexts = []
        labels = []
        L = len(words)
        for index, word in enumerate(words):
            s = index - window_size
            e = index + window_size + 1

            contexts.append([words[i] for i in range(s, e) if 0 <= i < L and i != index])
            labels.append(word)

            # pad_sequence: Pads each sequence to the same length (length of the longest sequence.
            # If maxlen is provided, any sequence longer than maxlen is truncated to maxlen. Truncation happens off
            # either the beginning or the end of the sequence.
            x = sequence.pad_sequences(contexts, maxlen=maxlen)
            # to_categorical: Converts a class vector to binary class matrix.
            y = np_utils.to_categorical(labels, V)

            # The yield statement
            # A yield statement is semantically equivalent to a yield expression. The yield statement can be used to
            # omit the parentheses that would otherwise be required in the equivalent yield expression statement.
            # For example, the yield statements.
            # 一个带有 yield 的函数就是一个 generator，它和普通函数不同，生成一个 generator 看起来像函数调用，
            # 但不会执行任何函数代码，直到对其调用 next()（在 for 循环中会自动调用 next()）才开始执行。
            # 虽然执行流程仍按函数的流程执行，但每执行到一个 yield 语句就会中断，并返回一个迭代值，
            # 下次执行时从 yield 的下一个语句继续执行。看起来就好像一个函数在正常执行的过程中被 yield 中断了数次，
            # 每次中断都会通过 yield 返回当前的迭代值。
            yield (x, y)


cbow = Sequential()
cbow.add(Embedding(input_dim=V, output_dim=dim, input_length=window_size * 2))
# Lambda: Wraps arbitrary expression as a Layer object.
# 本函数用以对上一层的输出施以任何Theano/TensorFlow表达式
# mean: Mean of a tensor, alongside the specified axis.
cbow.add(Lambda(lambda x: K.mean(x, axis=1), output_shape=(dim,)))
cbow.add(Dense(V, activation='softmax'))
cbow.compile(loss='categorical_crossentropy', optimizer='adadelta')
plot_model(cbow, to_file="model.png", show_shapes=True)

for ite in range(10):
    loss = 0.
    for x, y in generate_data(corpus, window_size, V):
        # train_on_batch: Single gradient update over one batch of samples.
        loss += cbow.train_on_batch(x, y)
    print(ite, loss)

f = open('vectors.txt', 'w')
f.write(' '.join([str(V - 1), str(dim)]))
f.write('\n')

vectors = cbow.get_weights()[0]
for word, i in tokenizer.word_index.items():
    f.write(word)
    f.write(' ')
    f.write(' '.join(map(str, list(vectors[i, :]))))
    f.write('\n')
f.close()

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
w2v.most_similar(positive=['alice'])
