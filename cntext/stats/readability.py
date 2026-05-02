import re
import jieba
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize, sent_tokenize
import warnings
warnings.filterwarnings('ignore')
import logging
jieba_logger = logging.getLogger('jieba')
jieba_logger.setLevel(logging.CRITICAL)



    


def cn_seg_sent(text):
    #split the chinese text into sentences
    text = re.sub(r'([。！；？;\?])([^”’])', "[[end]]", text)  # 单字符断句符
    text = re.sub(r'([。！？\?][”’])([^，。！？\?])', "[[end]]", text)
    text = re.sub(r'\s', '', text)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    return text.split("[[end]]")




def count_syllables(word):
    """计算单词的音节数"""
    word = word.lower()
    count = 0
    vowels = "aeiouy"
    if word[0] in vowels:
        count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
    if word.endswith("e"):
        count -= 1
    if count == 0:
        count += 1
    return count




def readability(text, lang='chinese', syllables=3, return_series=False):
    """
    计算文本可读性常见指标； 含Gunning Fog Index、 SMOG Index、Coleman Liau Index、 Automated Readability Index(ARI)、Readability Index(Rix)
    
    Gunning Fog Index = 0.4 * (Total_Words/Total_Sentences + 100 * Complex_Words/Total_Words)
    SMOG Index = 1.0430 * sqrt(Complex_Words/Total_Sentences) * 30 + 3.1291
    Coleman-Liau Index = 0.0588 * (100*Total_Letters/Total_Words) -0.296*(100*Total_Sentences/Total_Words) - 15.8
    Automated Readability Index(ARI) = 4.71 * (Total_Characters/Total_Words) + 0.5*(Total_Words/Total_Sentences) - 21.43
    Readability Index(RIX) = Complex_Words * (6 + Total_characters) / Total_Sentences


    Args:
        text (str): 待分析文本
        lang (str, optional): 设置text的语言类型， 'chinese'或'english', 默认'chinese'.
        syllables (int, optional): 音节数(汉字数)大于等于syllables为复杂词. 默认值为3
        return_series(boolean, optional): 计算结果是否输出为pd.Series类型，默认为False

    Returns:
        dict(or pd.Sereis): 多个可读性指标的字典
    """

    #总字符数
    total_characters = len(text)
    if lang=='chinese':
        #总字数
        total_letters = len(re.findall(r'[\u4e00-\u9fff]', text))
        
        sentences = cn_seg_sent(text)
        total_words = 0
        total_complex_words = 0
        total_sentences = len(sentences)
        for sentence in sentences:
            words = list(jieba.cut(sentence))
            for word in words:
                #每个汉字看做一个音节， 字节数大于等于syllables默认该词为多音节词(复杂词)
                if len(word) >= syllables:
                    total_complex_words += 1

    
    else:
        total_letters = sum(c.isalpha() for c in text)
        sentences = sent_tokenize(text)
        words = word_tokenize(text)
        total_words = len(words)
        total_sentences = len(sentences)
        #字节数大于等于syllables默认该词为多音节词(复杂词)
        total_complex_words = sum(1 for word in words if count_syllables(word) >= syllables)
    

    if total_sentences==0:
        total_sentences = 1
    if total_words==0:
        total_words = 1
        
    fog_index = 0.4 * (total_words/ total_sentences)  + 0.4 * (100* total_complex_words/ total_words)  
    flesch_kincaid_grade_level = 0.39 * (total_words/ total_sentences) + 11.8 * (total_complex_words/ total_words) - 15.59
    smog_index = 1.0430 * np.sqrt(total_complex_words/ total_sentences) * 30 + 3.1291
    coleman_liau_index = 0.0588 * (100 * total_letters / total_words)   - 0.296 * (100*total_sentences/total_words) - 15.8
    ari = 4.71 * (total_characters/ total_words) + 0.5*(total_words/ total_sentences) - 21.43
    rix = total_complex_words * (6 + total_characters) / total_sentences
    
    
    
    result = {'fog_index': round(fog_index, 2),
              'flesch_kincaid_grade_level': round(flesch_kincaid_grade_level, 2),
              'smog_index': round(smog_index, 2),
              'coleman_liau_index': round(coleman_liau_index, 2),
              'ari': round(ari, 2),
              'rix': round(rix, 2)}
    if return_series:
        return pd.Series(result)
    return result
        


