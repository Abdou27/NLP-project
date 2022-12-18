import re
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from typing import Union, List
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords


class Cleaner(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.stemmer = None

    def fit(self, X, y=None):
        return self

    def clean_text(self, text: str, tokenize: bool = True, exclude_words=None) -> Union[str, List[str]]:
        """
        Fonction de nettoyage d'un article de presse.

        :param text: Texte de l'article sous forme d'une string
        :param tokenize: Si True le résultat est renvoyé sous forme
        d'une liste de string sinon sous forme d'une string simple
        :param exclude_words: Mots sans valeur ajoutée à exclure
        :return: string ou liste de string du texte nettoyé
        """
        if exclude_words is None:
            exclude_words = {}
        # Passage en minuscules
        text = text.lower()

        # Suppression des codes HTML
        text = re.sub(r'&#[0-9]+;', '', text)

        # Suppression des slashs séparateurs de mots
        text = re.sub(r'(\w+)/(\w+)', r'\1 \2', text)

        # Suppression des sommes d'argent et conservation du symbole monétaire
        text = re.sub(r'\$[0-9]+([.][0-9]*)?(m|bn)?', '$', text)
        text = re.sub(r'£[0-9]+([.][0-9]*)?(m|bn)?', '£', text)

        # Suppression des tirets séparateurs de phrase
        text = text.replace(' - ', ' ')

        # Remplacement des espaces multiples par des espaces simples
        text = re.sub(r'\s+', ' ', text)

        # Suppression de la ponctuation restante
        text = re.sub(r'[.,;:!?"()\[\]{}]', '', text)

        # Découpage en liste de tokens
        tokens = word_tokenize(text)

        # Stemming = On cherche la racine des mots
        ps = PorterStemmer()
        # Mots courants ayant peu de valeur ajoutée
        stop_words = set(stopwords.words('english'))
        # Mots exclus sans valeur ajoutée
        stop_words = stop_words.union(set(exclude_words))
        tokens = [ps.stem(w) for w in tokens
                  # Suppression des stop words et mots exclus
                  if w not in stop_words]

        if not tokenize:
            return ' '.join(tokens)
        return tokens

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
        return X.apply(lambda x: self.clean_text(x, False))
