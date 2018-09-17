import re
import xml.sax.saxutils as saxutils

from bs4 import BeautifulSoup

from pandas import DataFrame

import warnings

from tfidf_classifier import TFIDFClassifier

import numpy as np

warnings.filterwarnings('ignore')


class Reuters():
    def __init__(self, data_root='./reuters21578/', topic_file_name='all-topics-strings.lc.txt'):
        self._data_root = data_root
        self._topic_file_name = topic_file_name
        self._number_of_files = 22
        self._file_name_prefix = 'reut2-'
        self._create_stat_template()
        self._data = DataFrame(columns=['Id', 'Topic', 'Set', 'Body', 'TFIDF'])

    def _create_stat_template(self):
        topics_stats = []
        with open(self._data_root + self._topic_file_name, 'r') as topic_file:
            for topic in topic_file.readlines():
                topic = topic.strip()
                topics_stats.append([topic, 'TRAIN', 0])
                topics_stats.append([topic, 'TEST', 0])
                topics_stats.append([topic, 'NOT-USED', 0])
                topics_stats.append([topic, 'USABLE', 0])

        self._article_stats = DataFrame(data=topics_stats, columns=['Topic', 'Set', 'Count'])

    def _remove_tags(self, text):
        return re.sub('<[^<]+?>', '', text).strip()

    def _update_stats_field(self, articles_stats, topic, set_class):
        idx = articles_stats[articles_stats.Topic == topic][articles_stats.Set == set_class].index[0]
        f = articles_stats.get_value(idx, 'Count')
        articles_stats.set_value(idx, 'Count', f + 1)

    def _update_stats(self, articles_stats, topic, set_class):
        self._update_stats_field(articles_stats, topic, set_class)
        if set_class in ['TEST', 'TRAIN']:
            self._update_stats_field(articles_stats, topic, 'USABLE')

    def _unescape(self, text):
        return saxutils.unescape(text)

    def _newslines(self):
        # for i in range(1):
        for i in range(self._number_of_files):
            file_id = '00' + str(i) if i < 10 else '0' + str(i)
            print("processing file {}".format(file_id))
            with open(self._data_root + self._file_name_prefix + file_id + '.sgm', 'r') as file:
                content = BeautifulSoup(file.read().lower())
                for newsline in content('reuters'):
                    yield newsline

    def _matrix_to_list(self, data):
        return [np.squeeze(np.asarray(item.todense())) for item in data]

    def get_news_stats(self) -> DataFrame:
        """
        returns stats of number of available news for each set
        :return datafram of stats with ['Topic', 'Set', 'Count'] as Columns
        """
        for newsline in self._newslines():
            set_class = newsline.attrs['lewissplit'].upper()
            topics = newsline.topics.contents
            for topic in topics:
                topic_cleaned = self._remove_tags(str(topic)).strip()
                self._update_stats(articles_stats=self._article_stats, topic=topic_cleaned, set_class=set_class)

        return self._article_stats

    def load_data(self):
        """
        Loads all the data from txt files to dataframe
        :return:
        """
        for newsline in self._newslines():
            document_id = newsline['newid']

            set_class = newsline.attrs['lewissplit'].upper()
            if set_class not in ['TRAIN', 'TEST']:
                continue

            # News text
            document_body = self._remove_tags(str(newsline('text')[0].text)).replace('reuter\n&#3;', '').replace('\t',
                                                                                                                 ' ')
            document_body = self._unescape(document_body.strip())

            # News topics
            topics = newsline.topics.contents
            for topic in topics:
                topic_cleaned = self._remove_tags(str(topic)).strip()
                self._data = self._data.append({
                    'Id': document_id,
                    'Topic': topic_cleaned,
                    'Set': set_class,
                    'Body': document_body,
                    'TFIDF': []
                }, ignore_index=True)

    def get_all_train(self, set: str):
        """
        Returen all TRAIN data to calculate TFIDF
        :param set:
        :return:
        """
        return self._data[self._data.Set == set].Body.values.tolist()

    def add_tfidf(self, tfidf_classifier: TFIDFClassifier):
        """
        Adds TFIDF to the dataframe to avoid multiple calculation of TFIDF
        :param tfidf_classifier: TFIDF classifier that can do word-to-tfidf conversion
        """
        document_matrix = tfidf_classifier.to_tfidf(self._data.Body.values.tolist())

        for index, row in enumerate(document_matrix):
            self._data.set_value(index, 'TFIDF', document_matrix[index])

    def get_data(self, topic: str, set: str):
        """
        getting vectorized TFIDF equivalent of news with belonging to specific topic and set
        :param topic: news topic. e.g. acq
        :param set: TRAIN or TEST
        :return: list of TFIDF in csr_matrix form
        """
        positive_examples = self._data[self._data.Topic == topic][self._data.Set == set].TFIDF.values.tolist()
        negative_examples = self._data[self._data.Topic != topic][self._data.Set == set].TFIDF.values.tolist()
        all_examples = self._matrix_to_list(positive_examples + negative_examples)

        labels = [0] * (len(positive_examples) + len(negative_examples))
        labels[0:len(positive_examples)] = [1] * len(positive_examples)
        return all_examples, labels
