# word2vec

Tool for computing continuous distributed representations of words.

## Introduction

This tool provides an efficient implementation of the continuous bag-of-words and skip-gram architectures for computing vector representations of words. These representations can be subsequently used in many natural language processing applications and for further research.

## Quick start

- Download the code: [svn checkout](http://word2vec.googlecode.com/svn/trunk/)
- Run 'make' to compile word2vec tool
- Run the demo scripts: `./demo-word.sh` and `./demo-phrases.sh`
- For questions about the toolkit, see [this](http://groups.google.com/group/word2vec-toolkit)

## How does it work

The **word2vec** tool takes a text corpus as input and produces the word vectors as output. It first constructs a vocabulary from the training text data and then learns vector representation of words. The resulting word vector file can be used as features in many natural language processing and machine learning applications.

A simple way to investigate the learned representations is to find to closest words for a user-specified word. The **distance** tool serves that purpose. For example, if you enter 'france', **distance** will display the most similar words and their distances to 'france', which should look like:

> ### Word Cosine distance
> spain 0.678515<br>
> belgium 0.665923<br>
> netherlands 0.652428<br>
> italy 0.633130<br>
> switzerland 0.622323<br>
> luxembourg 0.610033<br>
> portugal 0.577154<br>
> russia 0.571507<br>
> germany 0.563291<br>
> catalonia 0.534176<br>

There are two main learning algorithms in **word2vec**: continuous bag-of-words and continuous skip-gram. The switch **-cbow** allows the user to pick one of these learning algorithms. Both algorithms learn the reoresentation of a word that is useful for prediction of other words in the sentense. These algorithms are described in detail in `[1, 2]`.

## Interesting properties of the word vectors

It was recently shown that the word vectors capture many linguistic regularities, for example vector operations `vector('Paris') - vector('France') + vector('Italy')` results in a vector that is very close to `vector('Rome'), and vector('king') - vector('man') + vector('woman')` is close to `vector('queen') [3, 1]`. You can try out a simple demo by running `demo-analogy.sh`.

To observe strong regularitites in the word vector space, it is need to train the models on large data set, with sufficient vector dimensionality as shown in `[1]`. Using the **word2vec** tool, it is possible to train models on huge data sets (up to hundreds of billions of words).

## From words to phrases and beyond

In certain applications, it is useful to have vector representation of larger pieces of text. For example, it is desirable to have only one vector for representing 'san francisco'. This can be achieved by pre-processing the training data set to form the phrases using the **word2phrase** tool, as is shown in the example script `./demo-phrases.sh`. The example output with the closest takens to -san_francisco' looks like:

> ### Word Cosine distance
> los_angeles 0.666175<br>
> golden_gate 0.571522<br>
> oakland 0.557521<br>
> california 0.554623<br>
> san_diego 0.534939<br>
> pasadena 0.519115<br>
> seattle 0.512098<br>
> taiko 0.507570<br>
> houston 0.499762<br>
> chicago_illinois 0.491598<br>

The linearity of the vector operations seems to weakly hold also for the addition of several vectors, so it is possible to add several word of phrase vectors to form representation of short sentences.

## How to measure quality of the word vectors

Several factors influence the quality of the word vectors: 

- Amnout and quality of the training data
- Size of the vectors
- Training algorithm

The quality of the vectors is cruical for any application. However, exploration of different hyper-parameter settings for complex tasks might be too time demanding. Thus, we designed simple test that can used to quickly evaluate the word vector quality.

For the word relation test set descirbed in `[1]`, see `./demo-word-accuracy.sh`, for the phrase relation test set described in `[2]`, see `./demo-phrase-accuracy.sh`. Note that the accuracy depends heavily on the amount of the training data; our best results for both test sets are above 70% accuracy with coverage close to 100%.

## Word clustering

The word vectors can be also used for deriving word classes from huge data sets. This is achieved bu performing K-means clustering on top of the word vectors. The script that demonstrates this is `./demo-classes.sh`. The output is a vocabulary file with words and their corresponding class IDs, such as:

> carnivores 234 carnivorous 234 cetaceans 234 cormorant 234 coyotes 234 crocodile 234 crocodiles 234 crustaceans 234 cultivated 234 danios 234 . . . acceptance 412 argue 412 argues 412 arguing 412 argument 412 arguments 412 belief 412 believe 412 challenge 412 claim 412

## Performance

The training speed can be significantly improved by using parallel training on multiple-CPU machine (use the switch '-threads N'). The hyper-parameter choice is cruical for performance (both speed and accuracy), however varies for different applications. The main choieces to make are:

- architecture: skip-gram (slower, better for infrequent words)vs CBOW (fast)
- The training algorithm: hierarchical softmax (better for infrequent words) vs negative sampling (better for frequent words, better with low dimensional vectors)
- Sub-sampling of frequent words: can improve both accuracy and speed for large data sets (useful values are in range 1e-3 to 1e-5)
- Dimensionality of the word vectors: usually more is better, but not always
- Context (window) size: for skip-gram usually around 10, for CBOW around 5

## References

- `[1]` Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. Efficient Estimation of Word Representations in Vector Space. In Proceedings of Workshop at ICLR, 2013.

- `[2]` Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, and Jeffrey Dean. Distributed Representations of Words and Phrases and their Compositionality. In Proceedings of NIPS, 2013.

- `[3]` Tomas Mikolov, Wen-tau Yih, and Geoffrey Zweig. Linguistic Regularities in Continuous Space Word Representations. In Proceedings of NAACL HLT, 2013.