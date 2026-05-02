import re
import jieba
from collections import Counter
import pandas as pd
from ..io.dict import read_yaml_dict
from nltk.tokenize import word_tokenize
import logging
jieba_logger = logging.getLogger('jieba')
jieba_logger.setLevel(logging.CRITICAL)
import multiprocessing
import platform

STOPWORDS_zh = read_yaml_dict(yfile='enzh_common_StopWords.yaml')['Dictionary']['chinese']
STOPWORDS_en = read_yaml_dict(yfile='enzh_common_StopWords.yaml')['Dictionary']['english']




            


def cn_seg_sent(text):
    #split the chinese text into sentences
    text = re.sub(r'([。！；？;\?])([^”’])', "[[end]]", text)  # 单字符断句符
    text = re.sub(r'([。！？\?][”’])([^，。！？\?])', "[[end]]", text)
    text = re.sub(r'\s', '', text)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    return text.split("[[end]]")



def zh_split_sentences(text):
    """
    将中文文本分割为句子
    
    Args:
        text (str): 待分句的中文文本
        
    Returns:
        list
    """
    #split the chinese text into sentences
    # 1. 以句号分句
    text = re.sub(r'([。！；？;\?])([^”’])', "[[end]]", text)  # 单字符断句符
    text = re.sub(r'([。！？\?][”’])([^，。！？\?])', "[[end]]", text)
    text = re.sub(r'\s', '', text)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    return text.split("[[end]]")


def en_split_sentences(text):
    # 定义分句的正则表达式
    pattern = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s'
    # 根据正则表达式分句
    sentences = re.split(pattern, text)
    # 如果最后一个句子没有结束符，则加上一个空格和句号作为结束符
    if not sentences[-1].endswith((".", "!", "?")):
        sentences[-1] += "."
    # 去掉句子中的空格
    sentences = [sentence.strip() for sentence in sentences]
    # 去掉空句子
    sentences = list(filter(None, sentences))
    return sentences





def word_count(text, lang='chinese', return_df=False):
    """
    统计文本中的词频

    Args:
        text (str): 待分析的文本数据
        lang (str, optional): 文本的语言； 支持中英文，默认为chinese
        return_df (bool, optional): 返回结果是否为 dataframe . 默认False

    Returns:
        _type_: _description_
    """

    # remove punctuation
    if lang=='chinese':
        #text = ''.join(re.findall('[\u4e00-\u9fa5]+', text))
        words = list(jieba.cut(text))
        words = [w for w in words if w not in STOPWORDS_zh]
    else:
        words = text.lower().split(" ")
        words = [w for w in words if w not in STOPWORDS_en]
        
    if return_df:
        return pd.DataFrame(Counter(words).items(), columns=['word', 'freq'])
    else:
        return Counter(words)


        

def word_in_context(text, keywords, window=3, lang='chinese'):
    """
    在text中查找keywords出现的上下文内容(窗口window)，返回df。
    
    Args:
        text (str): 待分析文本
        keywords (list): 关键词列表
        window (int): 关键词上下文窗口大小
        lang (str, optional): 文本的语言类型， 中文chinese、英文english，默认chinese

    Returns:
        dataframe
    """
    if lang=='chinese':
        words = jieba.lcut(text.lower())
    elif lang=='english':
        try:
            words = word_tokenize(text.lower())
        except:
            warnings.warn("你应该安装nltk和对应的nltk_data, 请看B站https://www.bilibili.com/video/BV14A411i7DB")
            words = text.lower().split(' ')
    else:
        raise ValueError("lang参数只支持chinese和english")
            

    keywords = [w.lower() for w in keywords]
    kw_idxss = [[i for i, x in enumerate(words) if x == keyword] for keyword in keywords]
    rows = []
    for keyword, kw_idxs in zip(keywords, kw_idxss):
        for idx in kw_idxs:
            half = int((window-1)/2)
            start = max(0, idx - half)
            end = min(len(words), idx + half + 1)
            row = {'keyword': keyword, 
                   'context': ''.join(words[start: end]) if lang=='chinese' else ' '.join(words[start: end])
                      }
            rows.append(row)
    df = pd.DataFrame(rows)
    return df
