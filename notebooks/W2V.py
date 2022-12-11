from sklearn.base import BaseEstimator, TransformerMixin
from nltk import sent_tokenize
import numpy as np
from gensim.utils import simple_preprocess
import gensim.downloader as api


class W2V(BaseEstimator, TransformerMixin):
    def __init__(self, method: str = 'avg'):
        self.method = method
        self.wv = api.load('word2vec-google-news-300')

    def fit(self, X, y=None):
        return self

    def review_embeddings(self, review):
        """
        Return the Text vector using the average or sum of word embeddings given by Word2Vec
        """
        if self.method == 'avg':
            return np.mean([self.wv[word] for word in review if word in self.wv.index2word], axis=0)
        return np.sum([self.wv[word] for word in review if word in self.wv.index2word], axis=0)

    def transform(self, X, y=None):
        words = []
        for review in X.values.tolist():
            sent_rev = sent_tokenize(review)
            for word in sent_rev:
                words.append(simple_preprocess(word))
        return np.array([self.review_embeddings(review) for review in words])
