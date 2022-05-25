import re
import jieba
from collections import Counter
import numpy as np
import pathlib
import pickle
import os



def cn_seg_sent(text):
    #split the chinese text into sentences
    text = re.sub('([。！；？;\?])([^”’])', "[[end]]", text)  # 单字符断句符
    text = re.sub('([。！？\?][”’])([^，。！？\?])', "[[end]]", text)
    text = re.sub('\s', '', text)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    return text.split("[[end]]")


def load_pkl_dict(file, is_builtin=True):
    """
    load pkl dictionary file,
    :param file: pkl file path
    :param is_builtin: Whether it is the built-in pkl file of cntext. Default True
    """
    if is_builtin==True:
        pathchain = ['files',file]
        dict_file_path = pathlib.Path(__file__).parent.joinpath(*pathchain)
        dict_f = open(dict_file_path, 'rb')
    else:
        dict_f = open(file, 'rb')
    dict_obj = pickle.load(dict_f)
    return dict_obj




def dict_pkl_list():
    """
    Get the list of cntext built-in dictionaries (pkl format)
    """
    dict_file_path = pathlib.Path(__file__).parent.joinpath('files')
    return [f for f in os.listdir(dict_file_path) if 'pkl' in f]



STOPWORDS_zh = load_pkl_dict(file='STOPWORDS.pkl')['STOPWORDS']['chinese']
STOPWORDS_en = load_pkl_dict(file='STOPWORDS.pkl')['STOPWORDS']['english']
ADV_words = load_pkl_dict(file='ADV_CONJ.pkl')['ADV']
CONJ_words = load_pkl_dict(file='ADV_CONJ.pkl')['CONJ']



def term_freq(text, lang='chinese'):
    """
    Calculate word count
    :param text: text string
    :param language: "chinese" or "english"; default is "chinese"
    """
    if lang=='chinese':
        text = ''.join(re.findall('[\u4e00-\u9fa5]+', text))
        words = list(jieba.cut(text))
        words = [w for w in words if w not in STOPWORDS_zh]
    else:
        words = text.lower().split(" ")
        words = [w for w in words if w not in STOPWORDS_en]
    return Counter(words)



def readability(text, zh_adjconj=None, lang='chinese'):
    """
    text readability, the larger the indicator, the higher the complexity of the article and the worse the readability.
    :param text: text string
    :param zh_adjconj Chinese conjunctions and adverbs, receive list data type. By default, the built-in dictionary of cntext is used
    :param language: "chinese" or "english"; default is "chinese"
    ------------
    【English readability】english_readability = 4.71 x (characters/words) + 0.5 x (words/sentences) - 21.43；
    【Chinese readability】  Refer 【徐巍,姚振晔,陈冬华.中文年报可读性：衡量与检验[J].会计研究,2021(03):28-44.】
                 readability1  ---每个分句中的平均字数
                 readability2  ---每个句子中副词和连词所占的比例
                 readability3  ---参考Fog Index， readability3=(readability1+readability2)×0.5
                 以上三个指标越大，都说明文本的复杂程度越高，可读性越差。

    """
    if lang=='english':
        text = text.lower()
        #将浮点数、整数替换为num
        text = re.sub('\d+\.\d+|\.\d+', 'num', text)
        num_of_characters = len(text)
        #英文分词
        rgx = re.compile("(?:(?:[^a-zA-Z]+')|(?:'[^a-zA-Z]+))|(?:[^a-zA-Z']+)")
        num_of_words = len(re.split(rgx, text))
        #分句
        num_of_sentences = len(re.split('(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text))
        ari = (
                4.71 * (num_of_characters / num_of_words)
                + 0.5 * (num_of_words / num_of_sentences)
                - 21.43
        )
        return {"readability": ari}
    if lang=='chinese':
        if zh_adjconj:
            adv_conj_words = zh_adjconj
        else:
            adv_conj_words = set(ADV_words + CONJ_words)
        zi_num_per_sent = []
        adv_conj_ratio_per_sent = []
        text = re.sub('\d+\.\d+|\.\d+', 'num', text)
        #【分句】
        sentences = cn_seg_sent(text)
        for sent in sentences:
            adv_conj_num = 0
            zi_num_per_sent.append(len(sent))
            words = list(jieba.cut(sent))
            for w in words:
                if w in adv_conj_words:
                    adv_conj_num+=1
            adv_conj_ratio_per_sent.append(adv_conj_num/(len(words)+1))
        readability1 = np.mean(zi_num_per_sent)
        readability2 = np.mean(adv_conj_ratio_per_sent)
        readability3 = (readability1+readability2)*0.5
        return {'readability1': readability1,
                'readability2': readability2,
                'readability3': readability3}



def sentiment(text, diction, lang='chinese'):
    """
    calculate the occurrences of each sentiment category words in text;
    the complex influence of intensity adverbs and negative words on emotion is not considered,
    :param text:  text sring
    :param diction:  sentiment dictionary；
    :param lang: "chinese" or "english"; default lang="chinese"

    diction = {'category1':  'category1 emotion word list',
               'category2':  'category2 emotion word list',
               'category3':  'category3 emotion word list',
                ...
               }
    :return:
    """
    result_dict = dict()
    senti_categorys = diction.keys()

    stopword_num = 0
    for senti_category in senti_categorys:
        result_dict[senti_category+'_num'] = 0

    #sentence_num = len(re.split('[。！!？\?;；]+', text))-1

    
    if lang=='chinese':
        sentence_num = len(cn_seg_sent(text))
        words = list(jieba.cut(text))
        word_num = len(words)
        for word in words:
            if word in STOPWORDS_zh:
                stopword_num+=1
            for senti_category in senti_categorys:
                if word in diction[senti_category]:
                    result_dict[senti_category+'_num'] +=  1

    else:
        sentence_num = len(re.split('(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text))
        rgx = re.compile("(?:(?:[^a-zA-Z]+')|(?:'[^a-zA-Z]+))|(?:[^a-zA-Z']+)")
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
    return result_dict



def sentiment_by_valence(text, diction, lang='english'):
    """
    calculate the occurrences of each sentiment category words in text;
    the complex influence of intensity adverbs and negative words on emotion is not considered.
    :param text:  text sring
    :param diction:  sentiment dictionary with valence.；
    :param lang: "chinese" or "english"; default lang="english"

    :return:
    """
    score = 0

    def query_word(word, df):
        """
        query
        """
        try:
            return df[df["word"] == word]['valence'].values[0]
        except:
            return 0

    if lang == 'chinese':
        words = list(jieba.cut(text))
        for word in words:
            try:
                score += query_word(word=word, df=diction)
            except:
                score += 0

    else:
        rgx = re.compile("(?:(?:[^a-zA-Z]+')|(?:'[^a-zA-Z]+))|(?:[^a-zA-Z']+)")
        words = re.split(rgx, text)
        for word in words:
            try:
                score += query_word(word=word, df=diction)
            except:
                score += 0

    res = round(score/len(words), 2)
    return res





