import chardet
from importlib import resources
from opencc import OpenCC
import ftfy
import contractions
import re



def traditional2simple(text, mode='t2s'):
    """
    中文繁体 转 中文简体； 

    Args:
        text (str): 待转换的文本内容
        mode(str): 转换模式， 默认mode='t2s'繁转简; mode还支持s2t

    Returns:
        str: 返回转换成功的文本
    """
    
    cc = OpenCC(mode)  # 繁体2简体
    return cc.convert(text)





def get_cntext_path():
    """
    查看cntext的安装路径
    """
    with resources.path('cntext', '__init__.py') as p:
        return p.parent







def detect_encoding(file, num_lines=100):
    """
    诊断某文件的编码方式

    Args:
        file (str): 文件路径
        num_lines (int, optional):  默认读取文件前100行.

    Returns:
        encoding type
    """
    try:
        import cchardet as chardet
        with open(file, "rb") as f:
            msg = f.read()
            result = chardet.detect(msg)
            return result['encoding']
    except:
        detector = chardet.UniversalDetector()
        with open(file, 'rb') as f:
            for line in f:
                detector.feed(line)
                if detector.done:
                    break
                num_lines -= 1
                if num_lines == 0:
                    break
        detector.close()
        return detector.result['encoding']






def fix_text(text):
    """
    将不正常的、混乱编码的文本转化为正常的文本。
    :param text:
    :return:
    """
    return ftfy.fix_text(text)



def fix_contractions(text):
    """
    英文缩写处理函数， 如you're -> you are
    
    Args:
        text (str): 待分句的中文文本
        
    Returns:
        text(str)
    """
    return contractions.fix(text)


















# ==============================
# 预编译正则表达式 & 定义常量（语言无关）
# ==============================

# URL 模式（通用）
URL_PATTERN = re.compile(
    r'https?://[^\s]+|'
    r'www\.[^\s]+|'
    r'[a-zA-Z0-9][-a-zA-Z0-9]*\.(?:com|net|org|gov|cn|io|app|me|edu|mil)'
    r'[/\w\d\.\?\=\&\%\#\-\_\~\:\@\+]*',
    re.IGNORECASE
)

# 数字归一化模式（通用）
DIGIT_PATTERN = re.compile(r'\d+\.?\d*')

# ==============================
# 中文清洗专用常量
# ==============================
CHINESE_CHARS = r'\u4e00-\u9fff'
CHINESE_PUNCTUATION = r'，。！？；：""\'\'（）【】《》、·…—'
CHINESE_KEEP_PATTERN = re.compile(
    f'[^{CHINESE_CHARS}a-zA-Z0-9\\s{CHINESE_PUNCTUATION}]'
)

# ==============================
# 英文清洗专用常量
# ==============================
ENGLISH_PUNCTUATION = r'.,!?;:"\'()[]{}…-'
ENGLISH_KEEP_PATTERN = re.compile(
    f'[^a-zA-Z0-9\\s{ENGLISH_PUNCTUATION}]'
)


def clean_text(text: str, lang: str = "chinese") -> str:
    """
    根据指定语言对文本进行标准化清洗。
    
    处理步骤（通用）：
        1. 转为小写（英文归一化）
        2. 移除 URL 链接
        3. 归一化数字为统一占位符 "数字"（中文）或 "NUMBER"（英文）
        4. 清洗噪声字符（仅保留该语言允许的字符集）
    
    Args:
        text (str): 原始用户输入文本
        lang (str): 语言标识，支持 "chinese" (默认) 或 "english"
        
    Returns:
        str: 清洗后的标准化文本
    """
    if not isinstance(text, str):
        return ""

    # 步骤1: 英文转小写（归一化）
    cleaned = text.lower()

    # 步骤2: 移除 URL（优先处理，避免干扰后续规则）
    cleaned = URL_PATTERN.sub('', cleaned)

    # 步骤3: 数字归一化（根据语言选择占位符）
    number_placeholder = "数字" if lang == "chinese" else "NUMBER"
    cleaned = DIGIT_PATTERN.sub(number_placeholder, cleaned)

    # 步骤4: 移除噪声字符（根据语言选择保留规则）
    if lang == "chinese":
        cleaned = CHINESE_KEEP_PATTERN.sub('', cleaned)
        cleaned = re.sub(r'([\u4e00-\u9fff])\s+([\u4e00-\u9fff])', r'\1\2', cleaned)  # 中文词间去空格
    elif lang == "english":
        cleaned = ENGLISH_KEEP_PATTERN.sub('', cleaned)
        # 仅压缩“非中文字符之间的空格”
        cleaned = re.sub(r'([a-zA-Z])\s+([a-zA-Z])', r'\1 \2', cleaned)  # 保留英文词间空格
        
    else:
        raise ValueError(f"Unsupported language: {lang}. Supported: 'chinese', 'english'")
    # 压缩多个连续空格为单个空格
    cleaned = re.sub(r'\s+', ' ', cleaned)
    return cleaned.strip()