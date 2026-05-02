import jieba
import warnings
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import Differ, SequenceMatcher
import logging
jieba_logger = logging.getLogger('jieba')
jieba_logger.setLevel(logging.CRITICAL)


def transform(text1, text2, lang='chinese'):
    """
    文本向量化;

    Args:
        text1 (str): 文本
        text2 (str): 文本
        lang (str, optional): 文本语言, chinese or english. Defaults to 'chinese'.

    Returns:
        _type_: _description_
    """


    if lang=='chinese':
        text1 = ' '.join(list(jieba.cut(text1)))
        text2 = ' '.join(list(jieba.cut(text2)))
    elif lang=='english':
        text1 = text1.lower()
        text2 = text2.lower()
    else:
        raise ValueError('lang must be chinese or english.')

    corpus = [text1, text2]
    cv = CountVectorizer(binary=True)
    cv.fit(corpus)
    vec1 = cv.transform([text1]).toarray()
    vec2 = cv.transform([text2]).toarray()
    return text1, text2, vec1, vec2



def jaccard_sim(text1, text2, lang='chinese'):
    """
    jaccard文本相似度算法;
    
    Args:
       text1(str):  文本
       text2(str):  文本
       lang(str):  文本语言, chinese or english
       
    Return: 
    """
    text11, text22, vec1, vec2 = transform(text1, text2, lang=lang)
    """ returns the jaccard similarity between two lists """
    vec1 = set([idx for idx, v in enumerate(vec1[0]) if v > 0])
    vec2 = set([idx for idx, v in enumerate(vec2[0]) if v > 0])
    res =  len(vec1 & vec2) / len(vec1 | vec2)
    return format(res, '.2f')



def minedit_sim(text1, text2, lang='chinese'):
    """
    最小编辑距离(Minimum edit distance)文本相似度算法;
    
    Args:
        text1(str):  文本
        text2(str): 文本
        lang(str):  文本语言, chinese or english
    
    Returns:
    """
    
    if lang=='chinese':
        words1 = list(jieba.cut(text1))
        words2 = list(jieba.cut(text2))
    elif lang=='english':
        words1 = text1.lower().split()
        words2 = text2.lower().split()
    else:
        raise ValueError('lang must be chinese or english.')
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



def simple_sim(text1, text2, lang='chinese'):
    """
    simple文本相似度;
    
    参考 Microsoft Word 中的跟踪更改功能和 Unix 系统中的 diff 功能。
    该算法的基本思想是将文本分割成单词或字符，然后计算它们的差异。
    差异的数量除以文本的总长度，得到文本的相似度。
    
    Args:
       text1(str):  文本
       text2(str): 文本
       lang(str):  文本语言, chinese or english
       
    Returns:
    """
    if lang=='chinese':
        words1 = list(jieba.cut(text1))
        words2 = list(jieba.cut(text2))
    elif lang=='english':
        words1 = text1.lower().split()
        words2 = text2.lower().split()
    else:
        raise ValueError('lang must be chinese or english.')
    diff = Differ()
    diff_manipulate = list(diff.compare(words1, words2))
    c = len(diff_manipulate) / (len(words1) + len(words2))
    cmax = max([len(words1), len(words2)])
    res =  (cmax - c) / cmax
    return format(res, '.2f')



def cosine_sim(text1, text2, lang='chinese'):
    """
    cosine similarity algorithm;
    :param text1:  text string
    :param text2:  text string
    :param lang:  language, chinese or english
    :return:
    """
    text11, text22, vec1, vec2 = transform(text1, text2, lang=lang)
    res = cosine_similarity(vec1, vec2)[0][0]
    return format(res, '.2f')