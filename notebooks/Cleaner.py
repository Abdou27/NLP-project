import re
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.stem import WordNetLemmatizer


class Cleaner(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.stemmer = None

    def fit(self, X, y=None):
        return self

    def _transform(self, x):
        # Remove all the special characters
        document = re.sub(r'\W', ' ', str(x))

        # remove all single characters
        document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

        # Remove single characters from the start
        document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)

        # Substituting multiple spaces with single space
        document = re.sub(r'\s+', ' ', document, flags=re.I)

        # Removing prefixed 'b'
        document = re.sub(r'^b\s+', '', document)

        # Converting to Lowercase
        document = document.lower()

        # Lemmatisation
        document = document.split()

        document = [self.stemmer.lemmatize(word) for word in document]
        document = ' '.join(document)

        return document

    def transform(self, X, y=None):
        self.stemmer = WordNetLemmatizer()

        return X.apply(self._transform)
