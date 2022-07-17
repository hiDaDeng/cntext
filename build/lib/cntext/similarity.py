#!/usr/bin/python
# -*- coding: utf-8 -*-
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import Differ, SequenceMatcher
import jieba
#from math import *
import warnings

warnings.filterwarnings('ignore')


def __check_contain_chinese(check_str):
    """
    check language, it will be True when check_str is chinese character.
    :param check_str:
    :return:
    """
    for ch in check_str:
        if u'\u4e00' <= ch <= u'\u9fff':
            return True
    return False


def __transform(text1, text2):
    """
    Convert text1, text2 to vector vec1, vec2
    :param text1:
    :param text2:
    :return:
    """

    if __check_contain_chinese(text1):
        text1 = ' '.join(list(jieba.cut(text1)))
        text2 = ' '.join(list(jieba.cut(text2)))
    else:
        text1 = text1.lower()
        text2 = text2.lower()

    corpus = [text1, text2]
    cv = CountVectorizer(binary=True)
    cv.fit(corpus)
    vec1 = cv.transform([text1]).toarray()
    vec2 = cv.transform([text2]).toarray()
    return text1, text2, vec1, vec2


#def cosine_similarity(vec1, vec2):
#    cos_sim = cosine_similarity(vec1, vec2)[0][0]
#    return cos_sim[0][0]

def jaccard_sim(text1, text2):
    """
    jaccard similarity algorithm;
    :param text1:  text string
    :param text2:  text string
    :return:
    """
    text11, text22, vec1, vec2 = __transform(text1, text2)
    """ returns the jaccard similarity between two lists """
    vec1 = set([idx for idx, v in enumerate(vec1[0]) if v > 0])
    vec2 = set([idx for idx, v in enumerate(vec2[0]) if v > 0])
    res =  len(vec1 & vec2) / len(vec1 | vec2)
    return format(res, '.2f')

def minedit_sim(text1, text2):
    """
    Minimum edit distance similarity algorithm;
    :param text1:  text string
    :param text2: text string
    :return:
    """
    words1 = list(jieba.cut(text1.lower()))
    words2 = list(jieba.cut(text2.lower()))
    leven_cost = 0
    s = SequenceMatcher(None, words1, words2)
    for tag, i1, i2, j1, j2 in s.get_opcodes():
        if tag == 'replace':
            leven_cost += max(i2 - i1, j2 - j1)
        elif tag == 'insert':
            leven_cost += (j2 - j1)
        elif tag == 'delete':
            leven_cost += (i2 - i1)
    return format(leven_cost, '.2f')


def simple_sim(text1, text2):
    """
    simple similarity algorithm;
    Refer to the track change function in Microsoft Word and the diff function in Unix systems.
    :param text1:  text string
    :param text2: text string
    :return:
    """
    words1 = list(jieba.cut(text1.lower()))
    words2 = list(jieba.cut(text2.lower()))
    diff = Differ()
    diff_manipulate = list(diff.compare(words1, words2))
    c = len(diff_manipulate) / (len(words1) + len(words2))
    cmax = max([len(words1), len(words2)])
    res =  (cmax - c) / cmax
    return format(res, '.2f')


def cosine_sim(text1, text2):
    """
    cosine similarity algorithm;
    :param text1:  text string
    :param text2:  text string
    :return:
    """
    text11, text22, vec1, vec2 = __transform(text1, text2)
    res = cosine_similarity(vec1, vec2)[0][0]
    return format(res, '.2f')
