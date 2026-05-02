import pandas as pd
from collections import Counter
import re


# 提取重复代码到独立函数
def get_toc_range(text):
    try:
        toc_begin = [match.start() for match in re.finditer(r'\n目\s*录', text)][0]
        toc_end = toc_begin + 2000
    except IndexError:
        toc_begin = 0
        toc_end = 2500
    return toc_begin, toc_end

def get_kws_pattern(kws_pattern):
    if not kws_pattern:
        kws_pattern = '董事会报告|董事会报告与管理讨论|企业运营与管理评述|经营总结与分析|管理层评估与未来展望|董事局报告|管理层讨论与分析|经营情况讨论与分析|经营业绩分析|业务回顾与展望|公司经营分析|管理层评论与分析|执行摘要与业务回顾|业务运营分析'
    return kws_pattern

def get_chinese_number_maps():
    chinese_number_map = {'一': 1, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9, '十': 10}
    chinese_number_map2 = {1: '一', 2: '二', 3: '三', 4: '四', 5: '五', 6: '六', 7: '七', 8: '八', 9: '九', 10: '十', 11: '十一', 12: '十二'}
    return chinese_number_map, chinese_number_map2

# 修改 extract_mda 函数的异常处理
def extract_mda(text, kws_pattern=''):
    """
    从中国A股上市公司年报文本中提取MD&A内容(管理层讨论与分析)

    Args:
        text (str): 中国A股上市公司年报文本
        kws_pattern (str, optional): 使用kws_pattern定位MD&A的内容. 默认值kws_pattern=''.
        
        kws_pattern = '董事会报告|董事会报告与管理讨论|企业运营与管理评述|经营总结与分析|管理层评估与未来展望|董事局报告|管理层讨论与分析|经营情况讨论与分析|经营业绩分析|业务回顾与展望|公司经营分析|管理层评论与分析|执行摘要与业务回顾|业务运营分析'

    Returns:
        str: MD&A内容
    """
    try:
        mda = mda_func1(text, kws_pattern=kws_pattern)
    except Exception:
        try:
            mda = mda_func2(text, kws_pattern=kws_pattern)
        except Exception:
            mda = ''
    return mda

# 修改 mda_func1 函数
def mda_func1(text, kws_pattern=''):
    toc_begin, toc_end = get_toc_range(text)
    kws_pattern = get_kws_pattern(kws_pattern)

    toc_names = re.split(r'\n第[一二三四五六七八九十][节|章]', text[toc_begin: toc_end])[1:]
    toc_names = [re.sub(r'\.', '', t) for t in toc_names]
    toc_names = [re.sub(r'\d+', '', t).rstrip() for t in toc_names]
    toc_idxs = re.findall(r'\n第[一二三四五六七八九十][节|章]', text[toc_begin: toc_end])

    if toc_names + toc_idxs:
        toc_df = pd.DataFrame({'toc': toc_names})
        try:
            mda_toc_idx = toc_df[toc_df.toc.fillna('').str.contains(kws_pattern)].index[0]
        except IndexError:
            pass

        mda_toc_name = toc_idxs[mda_toc_idx].replace('\n', '') + toc_names[mda_toc_idx]
        mda_mask = re.sub(r'\d+|\. ⋯', '', mda_toc_name).rstrip()

        mda_next_name = toc_idxs[mda_toc_idx + 1].replace('\n', '') + toc_names[mda_toc_idx + 1]
        mda_next_mask = re.sub(r'\d+|\. ⋯', '', mda_next_name).rstrip()

        try:
            mda_mask1, mda_mask2 = re.split(r'\s', mda_mask)
            some_idxs = [match.start() for match in re.finditer(mda_mask2, text[toc_end:])]
            for idx in some_idxs:
                if re.findall(mda_mask1, text[toc_end:][idx - 50: idx + 50]):
                    mda_begin_idx = idx

            mda_next_mask1, mda_next_mask2 = re.split(r'\s', mda_next_mask)
            some_idxs2 = [match.start() for match in re.finditer(mda_next_mask2, text[toc_end:])]
            for idx in some_idxs2:
                if re.findall(mda_next_mask1, text[toc_end:][idx - 50: idx + 50]):
                    if idx > mda_begin_idx:
                        mda_end_idx = idx

            return text[toc_end:][mda_begin_idx:mda_end_idx]
        except Exception:
            raw_mda_text = mda_mask + text[toc_end:].split(mda_mask)[-1]
            return raw_mda_text.split(mda_next_mask)[0]
    else:
        chinese_number_map, chinese_number_map2 = get_chinese_number_maps()
        numbers = []
        for idx in [match.start() for match in re.finditer(kws_pattern, text)]:
            number_mask = re.findall('[一二三四五六七八九十]+', text[idx - 10:idx + 10])
            if number_mask:
                numbers.extend(number_mask)

        if numbers:
            idx_chinese_char = max(Counter(numbers))
            new_pattern = r'\n[第]*[一二三四五六七八九十]{1,}[节|章]*[,，:：、. \t]{1,5}' + kws_pattern
            mda_mask = [r.replace(r'\n', '') for r in re.findall(new_pattern, text)][-1]
            punct = re.sub(r'[\u4e00-\u9fa5]+', '', mda_mask)
            mda_next_mask = '\n' + chinese_number_map2[chinese_number_map[idx_chinese_char] + 1] + punct
            mda_content = text.split(mda_mask)[-1].split(mda_next_mask)[0]
            return mda_content
        return ''

# 修改 mda_func2 函数
def mda_func2(text, kws_pattern=''):
    toc_begin, toc_end = get_toc_range(text)
    kws_pattern = get_kws_pattern(kws_pattern)
    chinese_number_map, chinese_number_map2 = get_chinese_number_maps()

    numbers = []
    for idx in [match.start() for match in re.finditer(kws_pattern, text)]:
        number_mask = re.findall('[一二三四五六七八九十]+', text[idx - 10:idx + 10])
        if number_mask:
            numbers.extend(number_mask)

    if numbers:
        idx_chinese_char = max(Counter(numbers))
        new_kws_pattern = r'\n[第]*[一二三四五六七八九十]{1,}[节|章]*[,，:：、. \t]{1,5}' + kws_pattern
        mda_mask = [r.replace('\n', '') for r in re.findall(new_kws_pattern, text)][-1]
        punct = re.sub(r'[\u4e00-\u9fa5]+', '', mda_mask)
        mda_next_mask = '\n' + chinese_number_map2[chinese_number_map[idx_chinese_char] + 1] + punct
        mda_content = mda_mask + text.split(mda_mask)[-1].split(mda_next_mask)[0]
        return mda_content
    return ''