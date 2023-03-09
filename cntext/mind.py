import numpy as np
from numpy import dot
from numpy.linalg import norm
from time import time
from gensim.models.keyedvectors import KeyedVectors
import scipy.spatial.distance
import itertools

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
        try:
            self.model = KeyedVectors.load_word2vec_format(w2v_model_path, binary=False, no_header=True)
        except:
            self.model = KeyedVectors.load_word2vec_format(w2v_model_path)
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


    def divergent_association_task(words, minimum=7):
        """Compute DAT score, get the detail of algorithm, please refer to Olson, J. A., Nahas, J., Chmoulevitch, D., Cropper, S. J., & Webb, M. E. (2021). Naming unrelated words predicts creativity. Proceedings of the National Academy of Sciences, 118(25), e2022340118."""
        # Keep only valid unique words
        uniques = []
        for word in words:
            try:
                self.model.get_vector(word)
                uniques.append(word)
            except:
                pass
    

        # Keep subset of words
        if len(uniques) >= minimum:
            subset = uniques[:minimum]
        else:
            return None # Not enough valid words

        # Compute distances between each pair of words
        distances = []
        for word1, word2 in itertools.combinations(subset, 2):
            dist = scipy.spatial.distance.cosine(self.model.get_vector(word1), self.model.get_vector(word2))
            distances.append(dist)

        # Compute the DAT score (average semantic distance multiplied by 100)
        return (sum(distances) / len(distances)) * 100





# class Alignment(object):
#     def __init__(self):
#         pass



# def procrustes(A, B):
#     """
#     Learn the best rotation matrix to align matrix B to A
#     https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
#     """
#     # U, _, Vt = np.linalg.svd(B.dot(A.T))
#     U, _, Vt = np.linalg.svd(B.T.dot(A))
#     return U.dot(Vt)

# def intersect_vocab (idx1, idx2):
#     """ Intersect the two vocabularies

#     Parameters:
#     ===========
#     idx1 (dict): the mapping for vocabulary in the first group
#     idx2 (dict): the mapping for vocabulary in the second group

#     Returns:
#     ========
#     common_idx, common_iidx (tuple): the common mapping for vocabulary in both groups
#     """
#     common = idx1.keys() & idx2.keys()
#     common_vocab = [v for v in common]

#     common_idx, common_iidx = {v:i for i,v in enumerate (common_vocab)}, {i:v for i,v in enumerate (common_vocab)}
#     return common_vocab, (common_idx, common_iidx)

# def align_matrices (mat1, mat2, idx1, idx2):
#     """ Align the embedding matrices and their vocabularies.

#     Parameters:
#     ===========
#     mat1 (numpy.ndarray): embedding matrix for first group
#     mat2 (numpy.ndarray): embedding matrix for second group

#     index1 (dict): the mapping dictionary for first group
#     index2 (dict): the mapping dictionary for the second group

#     Returns:
#     ========
#     remapped_mat1 (numpy.ndarray): the aligned matrix for first group
#     remapped_mat2 (numpy.ndarray): the aligned matrix for second group
#     common_vocab (tuple): the mapping dictionaries for both the matrices
#     """  
#     common_vocab, (common_idx, common_iidx) = intersect_vocab (idx1, idx2)
#     row_nums1 = [idx1[v] for v in common_vocab]
#     row_nums2 = [idx2[v] for v in common_vocab]

#     #print (len(common_vocab), len (common_idx), len (common_iidx))
#     remapped_mat1 = mat1[row_nums1, :]
#     remapped_mat2 = mat2[row_nums2, :]
#     #print (mat1.shape, mat2.shape, remapped_mat1.shape, remapped_mat2.shape)
  
#     omega = procrustes (remapped_mat1, remapped_mat2)
#     #print (omega.shape)
#     # rotated_mat2 = np.dot (omega, remapped_mat2)
#     rotated_mat2 = np.dot (remapped_mat2, omega)

#     return remapped_mat1, rotated_mat2, (common_idx, common_iidx)






