import numpy as np
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Embedding, Lambda
from keras.utils.data_utils import get_file
from keras.utils import np_utils
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
import gensim

np.random.seed(13)

# path = get_file('alice.txt', origin='http://www.gutenberg.org/cache/epub/11/pg11.txt')
path = "test.txt"
corpus = open(path).readlines()[0:200]
corpus = [sentence for sentence in corpus if sentence.count(' ') >= 2]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)
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

            x = sequence.pad_sequences(contexts, maxlen=maxlen)
            y = np_utils.to_categorical(labels, V)

            yield (x, y)


cbow = Sequential()
cbow.add(Embedding(input_dim=V, output_dim=dim, input_length=window_size * 2))
cbow.add(Lambda(lambda x: K.mean(x, axis=1), output_shape=(dim,)))
cbow.add(Dense(V, activation='softmax'))
cbow.compile(loss='categorical_crossentropy', optimizer='adadelta')

for ite in range(10):
    loss = 0.
    for x, y in generate_data(corpus, window_size, V):
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
w2v = gensim.models.KeyedVectors.load_word2vec_format('./vectors.txt', binary=False)
w2v.most_similar(positive=['alice'])
