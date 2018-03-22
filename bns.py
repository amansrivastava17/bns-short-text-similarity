# coding=utf-8

from collections import Counter, namedtuple
from scipy.sparse import csr_matrix
from scipy.stats import norm
import numpy as np
import math

from nlp_utils import ngrams


class BNS:
    """Bi-normal Separation is a popular method to score textual data importance against its
    belonging category, it can efficiently find out important keywords in a document and assign
    a weighted positive score, also provide negative scoring for unimportant word for a document.

    Below are the description of variables used to calculate Bi-normal separation score for a
    word for each category (or classes).

    Features Descriptions:
    ======================
    pos = number of positive training cases, typically minority,
    neg = number of negative training cases,
    tp = number of positive training cases containing word,
    fp = number of negative training cases containing word,
    fn = pos - tp,
    tn = neg - fp,
    true positive rate(tpr) = P(word | positive class) = tp/pos
    false positive rate (fpr) = P(word | negative class) = fp/neg,
    Bi-Normal Separation (BNS): =  F-1(tpr)  –  F-1(fpr)
        (F-1) is  the  inverse  Normal  cumulative  distribution  function
    """

    def __init__(self, ngram_range=None):
        self.categories = []
        self.bound_min_score = 0.0005
        self.bound_max_score = 1 - 0.0005
        self.bns_scores = {}
        self.vectors = {}
        self.sentences_category_map = {}
        if ngram_range is None:
            self.ngram_range = [1, 2]

    def bound(self, value):
        """
        Bound the bns score under `bound_min_score` and `bound_max_score`
        Args:
            value (float): bnr score
        Returns:
            (float): bounded bnr score within min max limit
        """
        return max(self.bound_min_score, min(self.bound_max_score, value))

    @staticmethod
    def calculate_bns_score(tpr, fpr):
        """
        Calculate bns score for given `tpr` and `fpr` value
        Args:
            tpr (float): true positive rate
            fpr (float): false positive rate
        Returns:
            (float) : bns score
        """
        return norm.ppf(tpr) - norm.ppf(fpr)

    def get_bns_score(self, word, category):
        """
        Returns bns score for given `word` belongs to `category`
        Args:
            word (str): word whose bns score to be determined
            category (str): category or class in which word bns score has to be find
        Returns:
            score (float): bns score
        """
        score = None
        if word in self.bns_scores:
            if category in self.bns_scores[word]:
                score = self.bns_scores[word][category]
        return score

    @staticmethod
    def get_word_list(documents):
        """
        Given list of sentences
        Args:
            documents (list): list of documents
        Returns:
            words (set): set of unique of words in documents
        """
        words = []
        for doc in documents:
            words.extend(doc.split())
        return set(words)

    @staticmethod
    def get_word_count_in_category(documents, categories):
        """
        Create dict containing count of word for every category from document.
            Examples:
                documents - ['book cab', 'book me a taxi', 'book flight to mumbai']
                categories - ['book_cab', 'book_cab', 'book_flight']

                word_dict => {'book': {'book_cab': 2 , book_flight: 1}, 'cab': {'book_cab': 1},
                              'me': {'book_cab': 1}, 'a': {'book_cab':1 }, 'flight': {'book_flight': 1},
                              'to': {book_flight: 1}, 'mumbai': {'book_flight': 1}}
        Args:
            documents (list): list of documents
            categories (list): list of category for doc in documents

        Returns:
            word_dict (dict):  dict of word and their count in respective categories
        """
        word_dict = {}
        for sent, cat in zip(documents, categories):
            words = sent.split()
            for word in words:
                if word not in word_dict:
                    word_dict[word] = {cat: 1}
                else:
                    if cat not in word_dict[word]:
                        word_dict[word][cat] = 1
                    else:
                        word_dict[word][cat] += 1
        return word_dict

    def create_bns_score(self, documents, categories, word_category_count_dict):
        """
        Create a dict of words and their respective bns score for each categories
        Args:
            documents (list): list of documents
            categories (list): list category doc in documents
            word_category_count_dict (dict): dict containing word and their respective count in categories
        Returns:
            None
        """
        self.categories = list(set(categories))
        total_categories = len(categories)
        word_list = self.get_word_list(documents)

        for sent, cat in zip(documents, categories):
            if cat not in self.sentences_category_map:
                self.sentences_category_map[cat] = [sent]
            else:
                self.sentences_category_map[cat].append(sent)

        for index, word in enumerate(word_list):
            for category in self.categories:
                positive_sent = len(self.sentences_category_map[category])
                negative_sent = total_categories - positive_sent

                word_dict = word_category_count_dict[word]
                total_word_occurrence = sum(word_dict.values())
                if category in word_dict:
                    tp = word_dict[category]
                else:
                    tp = 0
                fp = total_word_occurrence - tp
                tpr = self.bound(tp / float(positive_sent))
                fpr = self.bound(fp / float(negative_sent))
                bns_score = self.calculate_bns_score(tpr, fpr)
                if not self.bns_scores.get(word, None):
                    self.bns_scores[word] = {'index': index, category: bns_score}
                else:
                    if not self.bns_scores.get(word, {}).get(category, None):
                        self.bns_scores[word][category] = bns_score

    def fit(self, training_documents, categories):
        """
        Fit the documents and categories to create bns vectors for documents
        Args:
            training_documents (list): list of documents
            categories (list): list category doc in documents
        Returns:
            None
        """
        word_category_count_dict = self.get_word_count_in_category(training_documents, categories)
        self.create_bns_score(training_documents, categories, word_category_count_dict)
        for category in self.sentences_category_map.keys():
            scores, indexes, counter = [], [], []
            for count, sentence in enumerate(self.sentences_category_map[category]):
                tokens = []
                for n in self.ngram_range:
                    tokens.extend(ngrams(sentence, n))
                tokens_dict = dict(Counter(tokens))
                for token, token_count in tokens_dict.iteritems():
                    token_meta_data = self.bns_scores.get(token, None)
                    if token_meta_data:
                        if category in token_meta_data:
                            scores.append(token_count * token_meta_data[category])
                            indexes.append(token_meta_data['index'])
                            counter.append(count)
            self.vectors[category] = csr_matrix((scores, (counter, indexes)),
                                                shape=(len(counter), len(self.bns_scores)))

    def transform(self, test_documents):
        """
        Return bns vectors for test documents
        Args:
            test_documents (list): list of documents to convert them to bns vectorizer
        Returns:
            test_vector (list): list of bns vectors for each doc in `test_documents`
        """
        test_vector = {}
        for category in self.categories:
            scores, indexes, counter = [], [], []
            for count, sentence in enumerate(test_documents):
                tokens = []
                for n in self.ngram_range:
                    tokens.extend(ngrams(sentence, n))
                tokens_dict = dict(Counter(tokens))
                for token, token_count in tokens_dict.iteritems():
                    token_meta_data = self.bns_scores.get(token, None)
                    if token_meta_data:
                        if category in token_meta_data:
                            scores.append(token_count * token_meta_data[category])
                            indexes.append(token_meta_data['index'])
                            counter.append(count)
            test_vector[category] = csr_matrix((scores, (counter, indexes)), shape=(len(counter), len(self.bns_scores)))
        return test_vector
