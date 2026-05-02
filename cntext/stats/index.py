import re
import jieba
from collections import Counter
import numpy as np
import pandas as pd
from ..io.dict import read_yaml_dict
import warnings
import string
import networkx as nx
from distinctiveness.dc import distinctiveness
import warnings
from ..stats.utils import STOPWORDS_zh, STOPWORDS_en,zh_split_sentences, en_split_sentences
warnings.filterwarnings('ignore')
import logging
jieba_logger = logging.getLogger('jieba')
jieba_logger.setLevel(logging.CRITICAL)






def fepu(text, ep_pattern='', u_pattern=''):
    """
    企业感知政策不确定性; 目前内置词典仅支持中文， 如要分析的文本是英文，需要设置ep_pattern和u_pattern参数。
    
    Li, Jing, Huihua Nie, Rui Ruan, and Xinyi Shen. "Subjective perception of economic policy uncertainty and corporate social responsibility: Evidence from China." International Review of Financial Analysis 91 (2024): 103022.


    Args:
        text (str): 某时期t某企业i的管理层讨论与分析md&a文本
        ep_pattern (str, optional): 字符串；经济政策类词典，用|间隔词语，形如 ep_pattern = ‘经济|金融|政策|治理|行政’
        u_pattern (str, optional): 字符串；不确定性词典，用|间隔词语，形如 u_pattern = ‘风险|危机|难以预测’



    Returns:
        pd.Series
    """
    import re
    import jieba
    import pandas as pd
    if (ep_pattern+u_pattern) == '':
        fepu_dict = read_yaml_dict('zh_common_FEPU.yaml')
        ep_pattern = '|'.join(fepu_dict['Dictionary']['经济政策'])
        u_pattern = '|'.join(fepu_dict['Dictionary']['不确定'])
        for key, words in fepu_dict['Dictionary'].items():
            [jieba.add_word(w, freq=20000) for w in words] 
    else:
        [jieba.add_word(w, freq=2000) for w in (ep_pattern+u_pattern).split('|')] 
        
    
    raw_df = pd.DataFrame({'sent': zh_split_sentences(text)})
    ep_mask = raw_df['sent'].fillna('').str.lower().str.contains(ep_pattern)
    u_mask = raw_df['sent'].fillna('').str.lower().contains(u_pattern)
    
    u_num = 0
    pu_text = ''.join(raw_df[ep_mask & u_mask]['sent'])
    pu_Counter = Counter(jieba.lcut(pu_text))
    for w, n in pu_Counter.items():
        if w in u_pattern:
            u_num+=n
    word_Num = len(jieba.lcut(''.join(re.findall('[\u4e00-\u9fa5]+', text))))+1
    
    result =  {'FEPUw': 100*u_num/word_Num,
               'FEPUs': 100*len(raw_df[ep_mask & u_mask]['sent'])/len(raw_df)}
    return pd.Series(result)



def epu(df, freq='Y', e_pattern='', p_pattern='', u_pattern=''):
    """
    经济政策不确定性； 目前内置词典仅支持中文， 如要分析的文本是英文，需要设置ep_pattern、u_pattern和u_pattern参数。
    
    Huang, Yun, and Paul Luk. "Measuring economic policy uncertainty in China." China Economic Review 59 (2020): 10136

    Args:
        df (DataFrame): 新闻数据; 格式为dataframe， 必须满足 df.columns==['date', 'text'] ; df中每一行一条新闻， df含所有时期所有的新闻.
        freq (str): 确定EPU指数的时间颗粒度； 如年Y, 月m, 日d, 默认 freq='Y'
        e_pattern (str, optional): 经济类词典，用|间隔词语，形如 e_pattern = ‘经济|金融’
        p_pattern (str, optional): 政策词典，用|间隔词语，形如 p_pattern = ‘政策|治理|行政’
        u_pattern (str, optional): 不确定性词典，用|间隔词语，形如 u_pattern = ‘风险|危机|难以预测’

    Returns:
        pd.Series: 
    """
    if (e_pattern+p_pattern+u_pattern) == '':
        epu_dict = read_yaml_dict('zh_common_EPU.yaml')
        e_pattern = '|'.join(epu_dict['Dictionary']['经济'])
        p_pattern = '|'.join(epu_dict['Dictionary']['政策'])
        u_pattern = '|'.join(epu_dict['Dictionary']['不确定'])
        for key, words in epu_dict['Dictionary'].items():
            [jieba.add_word(w, freq=20000) for w in words] 
    else:
        [jieba.add_word(w, freq=20000) for w in (e_pattern+p_pattern+u_pattern).split('|')] 
    
    df['date'] = pd.to_datetime(df['date'])
    
    datas = []
    for date, period_df in df.groupby(pd.Grouper(key='date', freq=freq)):
        data = dict()
        data['date'] = date #month是datetime型日期，一般为每个月的最后一日
        e_mask = period_df['text'].fillna('').str.lower().str.contains(e_pattern)
        p_mask = period_df['text'].fillna('').str.lower().str.contains(p_pattern)
        u_mask = period_df['text'].fillna('').str.lower().str.contains(u_pattern)

        #在出现经济词的新闻中，统计出现政策、不确定新的比例
        data['epu'] = (e_mask & p_mask & u_mask).sum() / e_mask.sum()
        datas.append(data)
    return pd.DataFrame(datas)
    


def semantic_brand_score(text, brands, lang='chinese', co_range=7, link_filter=2):
    """
    通过 SBS 来衡量品牌（个体、公司、品牌、关键词等）的重要性。


    
    Colladon, Andrea Fronzetti. "The semantic brand score." *Journal of Business Research* 88 (2018): 150-160.
    
    Args:
        text (string): 待分析文本
        brands (list): 词语（个体、公司、品牌、关键词等）列表；
        lang (str, optional): 默认语言为'chinese'.
        co_range (int, optional): 共现范围7
        link_filter (int, optional): 是否纳入共现的边的筛选条件， 默认2.

    Returns:
        dataframe:
    """
    #保证jieba能识别brands词。
    [jieba.add_word(w, freq=20000) for w in brands]


    regex = re.compile('[%s]' % re.escape(string.punctuation))
    if lang=='chinese':
        words = jieba.lcut(regex.sub('', text))
        words = [w for w in words if w not in STOPWORDS_zh]
    else:
        words = regex.sub(' ', text.lower()).split(' ')
        words = [w for w in words if w not in STOPWORDS_en]

        
    docs = [words]
    #Create a dictionary with frequency counts for each word
    countPR = Counter()
    for doc in docs:
        countPR.update(Counter(doc))
        
    #Calculate average score and standard deviation
    avgPR = np.mean(list(countPR.values()))
    stdPR = np.std(list(countPR.values()))

    PREVALENCE = {}
    for brand in brands:
        PREVALENCE[brand] = (countPR.get(brand, 0) - avgPR) / stdPR
        
        
        
    #Create an undirected Network Graph
    G = nx.Graph()

    #Each word is a network node
    nodes = set([word for doc in docs 
                for word in doc])
    G.add_nodes_from(nodes)

    #Add links based on co-occurrences
    for doc in docs:
        w_list = []
        length= len(doc)
        for k, w in enumerate(doc):
            #Define range, based on document length
            if (k+co_range) >= length:
                superior = length
            else:
                superior = k+co_range+1
            #Create the list of co-occurring words
            if k < length-1:
                for i in range(k+1,superior):
                    linked_word = doc[i].split()
                    w_list = w_list + linked_word
            #If the list is not empty, create the network links
            if w_list:    
                for p in w_list:
                    if G.has_edge(w,p):
                        G[w][p]['weight'] += 1
                    else:
                        G.add_edge(w, p, weight=1)
            w_list = []

    #Remove negligible co-occurrences based on a filter
    #Create a new Graph which has only links above
    #the minimum co-occurrence threshold
    G_filtered = nx.Graph() 
    G_filtered.add_nodes_from(G)
    for u,v,data in G.edges(data=True):
        if data['weight'] >= link_filter:
            G_filtered.add_edge(u, v, weight=data['weight'])

    #Optional removal of isolates
    isolates = set(nx.isolates(G_filtered))
    isolates -= set(brands)
    G_filtered.remove_nodes_from(isolates)
    
    
    
    #DIVERSITY
    #Calculate Distinctiveness Centrality
    DC = distinctiveness(G_filtered, normalize = False, alpha = 1)
    DIVERSITY_sequence=DC["D2"]

    #Calculate average score and standard deviation
    avgDI = np.mean(list(DIVERSITY_sequence.values()))
    stdDI = np.std(list(DIVERSITY_sequence.values()))
    #Calculate standardized Diversity for each brand
    DIVERSITY = {}
    for brand in brands:
        DIVERSITY[brand] = (DIVERSITY_sequence.get(brand, 0) - avgDI) / stdDI
        
        
    #Define inverse weights 
    for u,v,data in G_filtered.edges(data=True):
        if 'weight' in data and data['weight'] != 0:
            data['inverse'] = 1/data['weight']
        else:
            data['inverse'] = 1   

    #CONNECTIVITY
    CONNECTIVITY_sequence=nx.betweenness_centrality(G_filtered, normalized=False, weight ='inverse')
    #Calculate average score and standard deviation
    avgCO = np.mean(list(CONNECTIVITY_sequence.values()))
    stdCO = np.std(list(CONNECTIVITY_sequence.values()))
    #Calculate standardized Prevalence for each brand
    CONNECTIVITY = {}
    for brand in brands:
        CONNECTIVITY[brand] = (CONNECTIVITY_sequence.get(brand, 0) - avgCO) / stdCO
        
        
    #Obtain the Semantic Brand Score of each brand
    SBS = {}
    for brand in brands:
        SBS[brand] = PREVALENCE[brand] + DIVERSITY[brand] + CONNECTIVITY[brand]
        
    PREVALENCE_df = pd.DataFrame.from_dict(PREVALENCE, orient="index", columns = ["PREVALENCE"])
    DIVERSITY_df = pd.DataFrame.from_dict(DIVERSITY, orient="index", columns = ["DIVERSITY"])
    CONNECTIVITY_df = pd.DataFrame.from_dict(CONNECTIVITY, orient="index", columns = ["CONNECTIVITY"])
    SBS_df_ = pd.DataFrame.from_dict(SBS, orient="index", columns = ["SBS"])
    SBS_df = pd.concat([ PREVALENCE_df, DIVERSITY_df, CONNECTIVITY_df, SBS_df_ ], axis=1, sort=False)
    return SBS_df




def word_hhi(text, lang='chinese'):
    """
    计算文本词汇使用的HHI

    赫芬达尔-赫希曼指数(Herfindahl-Hirschman Index)**作为一种衡量市场集中度的经济指标，通常用于分析产业或市场中企业份额的分布情况。迁移到文本中，是否可以看做词汇多样性、语言表达多样性、语义丰富度代理变量。
    :param text:  待分析的文本
    :param lang:  待分析文本的语言， chinese或english
    """
    if lang=='chinese':
        words = jieba.lcut(text)
    else:
        words = text.lower().split(" ")
    word_counts = list(Counter(words).values())
    word_props = np.array(word_counts)/sum(word_counts)
    hhi_value = sum(w_prop**2 for w_prop in word_props)
    return hhi_value