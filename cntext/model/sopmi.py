from pathlib import Path
import jieba.posseg as pseg
import math, time
import jieba
import logging
# 在文件开头添加
jieba_logger = logging.getLogger('jieba')
jieba_logger.setLevel(logging.CRITICAL)


def read_seed_words(seed_file):
    try:
        with open(seed_file, encoding='utf-8') as f:
            return [line.strip().split('\t')[0] for line in f]
    except FileNotFoundError:
        print(f"Seed file {seed_file} not found.")
        return []

def preprocess_corpus(corpus_file, seed_file, lang='chinese'):
    sentiment_words = read_seed_words(seed_file)
    if lang == 'chinese':
        for word in sentiment_words:
            jieba.add_word(word, freq=20000)

    seg_data = []
    try:
        with open(corpus_file, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    seg_data.append([word.word for word in pseg.cut(line) if word.flag[0] not in ['u', 'w', 'x', 'p', 'q', 'm']])
    except FileNotFoundError:
        print(f"Corpus file {corpus_file} not found.")
    return seg_data

def calculate_cooccurrence(seg_data, seed_file, window_size=5):
    sentiment_words = read_seed_words(seed_file)
    cowords_list = []
    
    for sent in seg_data:
        if set(sentiment_words).intersection(set(sent)):
            for index, word in enumerate(sent):
                left = sent[max(0, index - window_size):index]
                right = sent[index: min(len(sent), index + window_size + 1)]
                context = left + right + [word]
                
                if set(sentiment_words).intersection(set(context)):
                    for i, w1 in enumerate(context):
                        if w1 in sentiment_words:
                            for w2 in context[i+1:]:
                                cowords_list.append(f"{w1}@{w2}")
    return cowords_list

def compute_pmi(seg_data, cowords_list, seed_file):
    def compute_mi(p1, p2, p12):
        return math.log2(p12) - math.log2(p1) - math.log2(p2)

    def collect_worddict(seg_data):
        word_dict = {}
        for line in seg_data:
            for word in line:
                word_dict[word] = word_dict.get(word, 0) + 1
        all = sum(word_dict.values())
        return word_dict, all

    def collect_cowordsdict(cowords_list):
        co_dict = {}
        candi_words = []
        for co_words in cowords_list:
            candi_words.extend(co_words.split('@'))
            co_dict[co_words] = co_dict.get(co_words, 0) + 1
        return co_dict, candi_words

    def collect_sentiwords(seed_file, word_dict):
        try:
            with open(seed_file, encoding='utf-8') as f:
                pos_words = set([line.strip().split('\t')[0] for line in f if
                                 line.strip().split('\t')[1] == 'pos']).intersection(set(word_dict.keys()))
                f.seek(0)
                neg_words = set([line.strip().split('\t')[0] for line in f if
                                 line.strip().split('\t')[1] == 'neg']).intersection(set(word_dict.keys()))
                return pos_words, neg_words
        except FileNotFoundError:
            print(f"Seed file {seed_file} not found.")
            return set(), set()

    word_dict, all = collect_worddict(seg_data)
    co_dict, candi_words = collect_cowordsdict(cowords_list)
    pos_words, neg_words = collect_sentiwords(seed_file, word_dict)

    pmi_dict = {}
    for candi_word in set(candi_words):
        pos_sum = 0.0
        neg_sum = 0.0
        
        for pos_word in pos_words:
            p1 = word_dict[pos_word] / all
            p2 = word_dict[candi_word] / all
            pair = pos_word + '@' + candi_word
            if pair in co_dict:
                p12 = co_dict[pair] / all
                pos_sum += compute_mi(p1, p2, p12)

        for neg_word in neg_words:
            p1 = word_dict[neg_word] / all
            p2 = word_dict[candi_word] / all
            pair = neg_word + '@' + candi_word
            if pair in co_dict:
                p12 = co_dict[pair] / all
                neg_sum += compute_mi(p1, p2, p12)

        so_pmi = pos_sum - neg_sum
        pmi_dict[candi_word] = so_pmi
    
    return pmi_dict

def save_results(pmi_dict, save_dir='SoPmi'):
    pos_dict = {}
    neg_dict = {}
    Path('output').mkdir(exist_ok=True)
    Path('output').joinpath('SoPMI').mkdir(exist_ok=True)
    negfile = Path('output').joinpath(save_dir, 'neg.txt')
    posfile = Path('output').joinpath(save_dir, 'pos.txt')

    try:
        with open(negfile, 'w+', encoding='utf-8') as f_neg, open(posfile, 'w+', encoding='utf-8') as f_pos:
            for word, word_score in pmi_dict.items():
                if word_score > 0:
                    pos_dict[word] = word_score
                else:
                    neg_dict[word] = abs(word_score)

            for word, pmi in sorted(pos_dict.items(), key=lambda x: x[1], reverse=True):
                f_pos.write(word + '\n')
            for word, pmi in sorted(neg_dict.items(), key=lambda x: x[1], reverse=True):
                f_neg.write(word + '\n')
    except FileNotFoundError:
        print(f"Error saving files in {save_dir}.")

def SoPmi(corpus_file, seed_file, lang='chinese', save_dir='SoPmi'):
    """_summary_

    Args:
        corpus_file (str): 语料txt文件的路径。utf-8编码。
        seed_file (str): 种子词txt文件路径。utf-8编码。
                         每行一个种子词，格式为：seed_word\tpos
                                               或
                                            seed_word\tneg
        lang (str, optional): 语料库的语言。默认为'chinese'。
        save_dir (str, optional):  结果存储到某个位置，默认'SoPmi'。
    """
    print('Step 1/4:...Preprocess   Corpus ...')
    start_time = time.time()
    seg_data = preprocess_corpus(corpus_file, seed_file, lang)

    print('Step 2/4:...Collect co-occurrency information ...')
    cowords_list = calculate_cooccurrence(seg_data, seed_file)

    print('Step 3/4:...Calculate   mutual information ...')
    pmi_dict = compute_pmi(seg_data, cowords_list, seed_file)

    print('Step 4/4:...Save    candidate words ...')
    save_results(pmi_dict, save_dir)
    
    end_time = time.time()
    duration = end_time - start_time
    duration = round(duration, 2)
    print('Finish! used {0} s'.format(duration))
    print(f'The candidate words are stored in output/{save_dir}')
