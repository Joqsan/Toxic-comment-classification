from collections import OrderedDict, Counter
from sklearn.base import TransformerMixin
from typing import List, Union
import numpy as np
from numpy.linalg import norm


class BoW(TransformerMixin):
    """
    Bag of words tranformer class
    
    check out:
    https://scikit-learn.org/stable/modules/generated/sklearn.base.TransformerMixin.html
    to know about TransformerMixin class
    """

    def __init__(self, k: int):
        """
        :param k: number of most frequent tokens to use
        """
        self.k = k
        # list of k most frequent tokens
        self.bow = None
        self.token_to_id = None

    def fit(self, X: np.ndarray, y=None):
        """
        :param X: array of texts to be trained on
        """
        # task: find up to self.k most frequent tokens in texts_train,
        # sort them by number of occurences (highest first)
        # store most frequent tokens in self.bow
        #raise NotImplementedError
        
        tokens_count = Counter(' '.join(X).split())
        self.bow, count_k_most = zip(*tokens_count.most_common(self.k))
        #self.bow = list(self.bow) + ['UNK']
        self.bow = list(self.bow)
        
        self.token_to_id = {token: idx for idx, token in enumerate(self.bow)}
        
        # fit method must always return self
        return self

    def _text_to_bow(self, text: str) -> np.ndarray:
        """
        convert text string to an array of token counts. Use self.bow.
        :param text: text to be transformed
        :return bow_feature: feature vector, made by bag of words
        """
        
        result = [0] * len(self.bow)
        
        #UNK_id = self.token_to_id['UNK']
        for token in text.split():
            if token in self.token_to_id:
                token_id = self.token_to_id[token]
                result[token_id] = result[token_id] + 1
            else:
                pass
        
        #UNK_id = self.token_to_id['UNK']
        
        #text_count = Counter(text.split())
        #for token, token_count in text_count.items():
        #    token_id = self.token_to_id.get(token, UNK_id)
        #    result[token_id] = result[token_id] + token_count
        #raise NotImplementedError
        return np.array(result, "float32")

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        :param X: array of texts to transform
        :return: array of transformed texts
        """
        assert self.bow is not None
        return np.stack([self._text_to_bow(text) for text in X])

    def get_vocabulary(self) -> Union[List[str], None]:
        return self.bow


class TfIdf(TransformerMixin):
    """
    Tf-Idf tranformer class
    if you have troubles implementing Tf-Idf, check out:
    https://streamsql.io/blog/tf-idf-from-scratch
    """

    def __init__(self, k: int = None, normalize: bool = False):
        """
        :param k: number of most frequent tokens to use
        if set k equals None, than all words in train must be considered
        :param normalize: if True, you must normalize each data sample
        after computing tf-idf features
        """
        self.k = k
        self.normalize = normalize
        self.i = 0
        
        self.token_to_id = None
        
        # self.idf[term] = log(total # of documents / # of documents with term in it)
        #self.idf = OrderedDict()
        self.idf = None

    def fit(self, X: np.ndarray, y=None):
        """
        :param X: array of texts to be trained on
        """
        #raise NotImplementedError
        
        # 1. Find k most frequent tokens
        tokens_count = Counter(' '.join(X).split())
        token_k_most, count_k_most = zip(*tokens_count.most_common(self.k))
        #self.bow = list(self.bow) + ['UNK']
        token_k_most = list(token_k_most)
        
        self.token_to_id = {token: idx for idx, token in enumerate(token_k_most)}
        
        # 2. Learn the idf vector
        counter = np.zeros((len(X), len(self.token_to_id)))
        
        # 2.1. Equivalent to CountVectorizer
        for i, text in enumerate(X):
            text_count = Counter(text.split())
            for token, token_count in text_count.items():
                if token in self.token_to_id:
                    counter[i, self.token_to_id[token]] = token_count
                else:
                    pass
        
        # 2.2. Find the idf values
        df_t = (counter > 0).sum(axis=0)
        #idf = np.log((len(X)) / df_t)
        #idf = np.log((len(X) + 1) / (df_t + 1)) + 1
        self.idf = np.log((len(X) + 1) / (df_t + 1)) + 1
        
        # 2.3. Pass to self.idf
        #for term, idx in self.token_to_id.items():
        #    self.idf.update({term: idf[idx]})
        
        # fit method must always return self
        return self

    def _text_to_tf_idf(self, text: str) -> np.ndarray:
        """
        convert text string to an array tf-idfs.
        *Note* don't forget to normalize, when self.normalize == True
        :param text: text to be transformed
        :return tf_idf: tf-idf features
        """
        
        counter = np.zeros(len(self.token_to_id))
        
        # 2.1. Equivalent to CountVectorizer
        text_count = Counter(text.split())
        for token, token_count in text_count.items():
            if token in self.token_to_id:
                counter[self.token_to_id[token]] = token_count
            else:
                pass
        
        #print(self.i)
        #self.i += 1
        #print(counter)
            
        tf = counter / counter.sum() if counter.sum() else counter
        
        result = tf * self.idf
        
        if self.normalize:
            result = result / norm(result) if norm(result) else result

        #raise NotImplementedError
        return result

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        :param X: array of texts to transform
        :return: array of transformed texts
        """
        assert self.idf is not None
        return np.stack([self._text_to_tf_idf(text) for text in X])
