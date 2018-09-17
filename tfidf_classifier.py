from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
import re
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix

cachedStopWords = stopwords.words("english")


class TFIDFClassifier():
    def __init__(self, all_data):
        self._initilalize_tfidf(all_data)
        self._clf = CalibratedClassifierCV(base_estimator=LinearSVC(penalty='l2', dual=False, verbose=True), cv=5)

    def _initilalize_tfidf(self, train_data):
        tfidf = TfidfVectorizer(tokenizer=self._tokenize, min_df=3,
                                max_df=0.90, max_features=3000,
                                use_idf=True, sublinear_tf=True,
                                norm='l2');
        tfidf.fit(train_data)
        self._tfidf_transformer = tfidf

    def to_tfidf(self, text: list):
        """
        convers list of articles text to TFIDF
        :param text: List of articles text
        :return: document matrix of TFIDF
        """
        return self._tfidf_transformer.transform(text)

    def train(self, data: list, labels: list):
        """
        trains classifier on given data and labels
        :param data: document matrix of TFIDF
        :param labels: corresponding correct labels
        """
        self._clf.fit(data, labels)

    def predict(self, data: list)->list:
        return self._clf.predict(data)

    def get_precision_recall(self, data: list, labels: list):
        """
        return tuple of precision, recall on given data and labels
        :param data: document matrix of TFIDF
        :param labels: corresponding correct labels
        :return: tuple (precision, recall)
        """
        predicted_y = self._clf.predict(data)
        tn, fp, fn, tp = confusion_matrix(labels, predicted_y).ravel()
        precision_score = tp / (tp + fp)
        recall_score = tp / (tp + fn)
        return precision_score, recall_score

    def _tokenize(self, text):
        min_length = 3
        words = map(lambda word: word.lower(), word_tokenize(text))
        words = [word for word in words if word not in cachedStopWords]
        tokens = (list(map(lambda token: PorterStemmer().stem(token), words)))
        # Discarding tokens with characters except [a-zA-Z]
        p = re.compile('[a-zA-Z]+')
        filtered_tokens = list(filter(lambda token: p.match(token) and len(token) >= min_length, tokens))
        return filtered_tokens
