import numpy as np
from numpy import dot
from numpy.linalg import norm
from time import time
from gensim.models.keyedvectors import KeyedVectors


class Text2Mind(object):
    """
    Calculate cognitive (attitude, bias) direction and strength in text
    """
    def __init__(self, w2v_model_path='glove_w2v.6B.100d.txt'):
        """
        Init the Text2Mind
        :param w2v_model_path:  pretained embedding model file path, only support word2vec format pretained model！
        """
        print('Loading the model of {}'.format(w2v_model_path))
        start = time()
        
        self.model = KeyedVectors.load_word2vec_format(w2v_model_path,
                                                       binary=False,
                                                       unicode_errors='ignore')
        duration = round(time()-start, 2)
        print('Load successfully, used {} s'.format(duration))

    def k2v_model(self, word):
        """
        return the KeyedVectors object.
        """
        return self.model


    def get_vector(self, word):
        """
        get word vector
        """
        return self.model.get_vector(word)


    def __get_centroid(self, words):
        """
        calculate the centroid vector of multiple word vectors
        :param words: word list
        :return:
        """
        container = np.zeros(self.model.vector_size)
        for word in words:
            try:
                container = container + self.model.get_vector(word)
            except:
                assert "No word in embeddings models"
        return container / len(words)


    def sematic_projection(self, words, c_words1, c_words2):
        """
        Calculate the projected length of each word in the concept vector.Note that the calculation result reflects the direction of concept.
        Greater than 0 means semantically closer to c_words2.

        Refer "Grand, G., Blank, I.A., Pereira, F. and Fedorenko, E., 2022. Semantic projection recovers rich human knowledge of multiple object features from word embeddings. _Nature Human Behaviour_, pp.1-13."

        For example, in the size concept, if you want positive means big, and negative means small,
        you should set c_words1 = ["small", "little", "tiny"] c_words2 = ["large", "big", "huge"].

        :param words: words list
        :param c_words1: concept words1, c_words1 = ["small", "little", "tiny"]
        :param c_words2: concept words2, c_words2 = ["large", "big", "huge"]
        :param c_vector: concept_vector; the result of .build_concept(c_words1, c_words2)
        :return:
        """
        container = []
        source_vector = self.__get_centroid(c_words1)
        target_vector = self.__get_centroid(c_words2)
        c_vector = target_vector - source_vector
        concept_norm = norm(c_vector)
        for word in words:
            any_vector = self.model.get_vector(word)
            projection_score = dot(any_vector, c_vector) / concept_norm
            projection_score = round(projection_score, 2)
            container.append((word, projection_score))
        container = sorted(container, key=lambda k: k[1], reverse=False)
        return container


    def sematic_distance(self, words, c_words1, c_words2):
        """
        Calculate the distance from words with c_words1 and c_words2 respectively, and return the difference between the two distance.
        Greater than 0 means semantically closer to c_words2

        :param words: words list, words = ['program', 'software', 'computer']
        :param c_words1: concept words1, c_words1 = ["man", "he", "him"]
        :param c_words2: concept words2, c_words2 = ["woman", "she", "her"]
        :return:
        """
        any_vector = self.__get_centroid(words)
        c_vector1 = self.__get_centroid(c_words1)
        c_vector2 = self.__get_centroid(c_words2)
        dist_1 = np.linalg.norm(any_vector - c_vector1)
        dist_2 = np.linalg.norm(any_vector - c_vector2)
        res = dist_1-dist_2
        return round(res, 2)







