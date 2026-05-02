"""
使用csv数据，其中文本字段名可能是doc、content、text。

通过读取csv, 预处理中英文(可参考model.utils.preprocess_line）， 自动寻找最佳的topic数，实现LDA、动态主题模型​。
"""




#from ..topic.utils import preprocess_line
#import tomotopy as tp
#import pandas as pd


#def LDA(csvf, label, text, k=None, lang='chinese', stopwords_file=None, **kwargs):
#    df = pd.read_csv(csvf)
#    docs = df[text].apply(lambda x: preprocess_line(x, lang=lang, #stopwords=stopwords_file))
#    model = tp.DAModel(k=k)
#    for doc in docs:
#        model.add_doc(doc)
    


#preprocess_line(line, lang='chinese', stopwords=None)





