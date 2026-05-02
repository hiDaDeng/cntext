import re
import jieba
import pandas as pd
from nltk.tokenize import word_tokenize, sent_tokenize
from ..stats.utils import STOPWORDS_zh, STOPWORDS_en, zh_split_sentences
import logging
jieba_logger = logging.getLogger('jieba')
jieba_logger.setLevel(logging.CRITICAL)


def sentiment(text, diction, lang='chinese', return_series=False):
    """
    常见的情感分析默认情绪词无(等)权重， 通过统计词语个数来反应情感信息。
    
    Args:
        text (str):  待分析的文本字符串
        diction (dict): 格式为Python字典类型。    参数diction样式
    
        diction = {'category1':  'category1 emotion word list',
                  'category2':  'category2 emotion word list',
                  'category3':  'category3 emotion word list'}
                
        lang (str, optional): 文本的语言类型， 中文chinese、英文english，默认chinese。
        return_series(boolean, optional): 计算结果是否输出为pd.Series类型，默认为False

    Returns:
        dict(or pd.Series)
    """
    
     
    result_dict = dict()
    senti_categorys = diction.keys()

    stopword_num = 0
    for senti_category in senti_categorys:
        result_dict[senti_category+'_num'] = 0

    #sentence_num = len(re.split('[。！!？\?;；]+', text))-1
 
    if lang=='chinese':
        # using add_word to add chinese word in jieba
        for senti_category in senti_categorys:
            senti_category_words = diction[senti_category]
            for w in senti_category_words:
                try:
                    jieba.add_word(w, freq=20000)
                except:
                    pass


        sentence_num = len(zh_split_sentences(text))
        words = list(jieba.cut(text))
        word_num = len(words)
        for word in words:
            if word in STOPWORDS_zh:
                stopword_num+=1
            for senti_category in senti_categorys:
                if word in diction[senti_category]:
                    result_dict[senti_category+'_num'] +=  1

    else:
        sentence_num = len(re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text.lower()))
        rgx = re.compile(r"(?:(?:[^a-zA-Z]+')|(?:'[^a-zA-Z]+))|(?:[^a-zA-Z']+)")
        words = re.split(rgx, text)
        word_num = len(words)

        for word in words:
            if word in STOPWORDS_en:
                stopword_num+=1
            for senti_category in senti_categorys:
                if word in diction[senti_category]:
                    result_dict[senti_category+'_num'] +=  1



    result_dict['stopword_num'] = stopword_num
    result_dict['word_num'] = word_num
    result_dict['sentence_num'] = sentence_num
    if return_series:
        return pd.Series(result_dict)
    return result_dict


    


def sentiment_by_valence(text, diction, lang='chinese', mean=False, return_series=False):
    """
    Calculate the occurrences of each sentiment category words in text;
    the complex influence of intensity adverbs and negative words on emotion is not considered.
    
    Args:
        text (str): 待分析的文本字符串
        diction (dict): 格式为Python字典类型
        lang (str, optional):文本的语言类型， 中文chinese、英文english，默认chinese。
        mean (boolean, optional): 是否基于词语数量统计情绪信息， 默认False. 
        return_series(boolean, optional): 计算结果是否输出为pd.Series类型，默认为False

    Returns:
        dict(or pd.Series)
    """

    result = dict()
    attrs = pd.DataFrame(diction).index
    for attr in attrs:
        result[attr] = 0
    

    if lang == 'chinese':
        words = list(jieba.cut(text))
        for word in words:
            if diction.get(word):
                for attr in attrs:
                    result[attr] = result[attr] + diction.get(word)[attr]


    else:
        text = text.lower()
        #rgx = re.compile("(?:(?:[^a-zA-Z]+')|(?:'[^a-zA-Z]+))|(?:[^a-zA-Z']+)")
        #words = re.split(rgx, text)
        try:
            words = word_tokenize(text)
        except:
            #print('你的电脑nltk没配置好，请观看视频https://www.bilibili.com/video/BV14A411i7DB')
            rgx = re.compile(r"(?:(?:[^a-zA-Z]+')|(?:'[^a-zA-Z]+))|(?:[^a-zA-Z']+)")
            words = re.split(rgx, text)

        for word in words:
            if diction.get(word):
                for attr in attrs:
                    result[attr] = result[attr] + diction.get(word)[attr]
      
    result['word_num'] = len(words)
    
    if mean==True:
        for attr in attrs:
            result[attr] = round(result[attr]/len(words), 3)
    if return_series:
        return pd.Series(result)
    return result 