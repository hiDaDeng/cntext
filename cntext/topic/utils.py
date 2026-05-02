
import re
import jieba
from nltk.tokenize import word_tokenize

   
#常见符号
punctuation = '＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､\u3000、〃〈〉《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏﹑﹔·．！？｡。'

def preprocess_line(line, lang='chinese', stopwords=None):
    """
    根据语言对单行文本进行预处理。
    
    Args:
        line (str): 输入的一行文本。
        lang (str): 语言类型，支持 'chinese' 或 'english'。
        stopwords (set): 停用词集合。
        
    Returns:
        list: 预处理后的分词结果。
    """
    # 去除首尾空格并检查空行
    line = line.strip()
    if not line:
        return []
    
    # 将停用词转换为frozenset
    stopwords = frozenset(stopwords) if stopwords else frozenset()

    if lang == 'chinese':
        line = re.sub(f'[^{punctuation}\u4e00-\u9fa5a-zA-Z0-9]', '', line.lower())
        # 使用生成器表达式优化内存
        tokens = [word 
                  for word in (w.strip() for w in jieba.cut(line))
                  if word and word not in stopwords and len(word)>1]
    elif lang == 'english':
        # 使用生成器表达式优化内存
        line = re.sub(f'[^{punctuation}a-zA-Z0-9]', '', line.lower())
        tokens = [word for word in (w.strip() for w in word_tokenize(line))
                 if word and word not in stopwords]
    else:
        raise ValueError(f"Unsupported language: {lang}")

    return tokens