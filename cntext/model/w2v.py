from pathlib import Path
import time
from ..model.utils import preprocess_line,  get_optimal_threshold
from tqdm import tqdm
from gensim.models import word2vec
from gensim.models.word2vec import LineSentence
import gc
import psutil
from tqdm import tqdm
from gensim.models import Phrases
import smart_open
from functools import partial
from multiprocessing import Pool, cpu_count
import jieba
import logging
jieba_logger = logging.getLogger('jieba')
jieba_logger.setLevel(logging.CRITICAL)



# 在Word2Vec函数定义之前添加
def process_line(line, lang, stopwords):
    """处理单行文本的函数"""
    return preprocess_line(line, lang=lang, stopwords=stopwords)


def Word2Vec(corpus_file, lang='chinese', dict_file=None, stopwords_file=None, vector_size=100, window_size=6, min_count=5, max_iter=5, chunksize=10000, only_binary=True, **kwargs):
    """
    可直接对原始语料txt文件进行自动Word2vec训练。该函数会自动处理文本预处理(分词、去停词)、内存管理、参数调整等问题，确保训练过程顺利进行。

    Args:
        corpus_file (str): 语料库txt文件的路径。utf-8编码。
        lang (str, optional): 语料库的语言。默认为'chinese'。
        dict_file (str): 自定义词典txt文件路径，默认为None。utf-8编码。
        stopwords_file (str, optional): 停用词文件路径，默认为 None。
        vector_size (int, optional): 词向量的维度。默认为100。
        window_size (int, optional): 窗口大小。默认为10。
        min_count (int, optional): 最小词频。默认为5。
        max_iter (int, optional):   最大迭代次数。默认为5。
        chunksize (int, optional): 每次读取的行数。默认为10000。越大速度越快。
        only_binary (bool, optional): 是否只保存模型为二进制文件， 默认为True，只保存bin。 False时保存为txt和bin。
        **kwargs: 其他gensim可选参数，如negative、sample、hs等。
        
        
    Returns:
        gensim.models.keyedvectors.KeyedVectors: 训练好的Word2Vec模型。
    """
    
    start  = time.time()
    

    # 加载用户词典和停用词
    jieba.enable_parallel(cpu_count())
    if dict_file:
        jieba.load_userdict(dict_file)
    
    stopwords = set()
    if stopwords_file:
        try:
            with open(stopwords_file, 'r', encoding='utf-8') as f:
                stopwords = set([line.strip() for line in f if line.strip()])
        except Exception as e:
            print(f"Warning: Failed to load stopwords file: {e}")
            

    
    # 设置路径
    output_dir = Path('output')
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_name = Path(corpus_file).stem  # 去除文件扩展名
    cache_corpus_txt_file = output_dir / f"{corpus_name}_cache.txt"
    max_worker = int(cpu_count()*0.8)
    

    memory_threshold = get_optimal_threshold()
    
    # 检查缓存文件是否存在且不为空
    if cache_corpus_txt_file.exists() and cache_corpus_txt_file.stat().st_size > 0:
        print(f"Cache {cache_corpus_txt_file} Found, Skip Preprocessing Corpus")
        
    else:
        print(f"Cache {cache_corpus_txt_file} Not Found or Empty, Preprocessing Corpus")
        
        # 预处理语料
        with smart_open.open(corpus_file, 'r', encoding='utf-8') as lines_f:
            total_lines = sum(1 for _ in lines_f)  # 获取总行数
            # 动态计算buffer_size
            corpus_size = Path(corpus_file).stat().st_size  # 获取语料文件大小
            
            if corpus_size > 50 * 1024 * 1024 and total_lines==1:
                raise ValueError(r'The file contains no newline characters. It is recommended to preprocess the file to include multiple "\n" characters, enabling line-by-line processing to avoid memory overflow.')

            lines_f.seek(0)  # 添加这一行

            with smart_open.open(cache_corpus_txt_file, 'w', encoding='utf-8') as cache_f:
                # 使用partial固定lang和stopwords参数
                process_func = partial(process_line, lang=lang, stopwords=stopwords)
                
                with Pool(processes=max_worker) as pool:
                    buffer_size = chunksize * 10
                    check_interval = chunksize * 10
                    processed_lines = 0

                    buffer = [] # 根据内存情况动态调整缓冲区大小
                    try:
                        for words in tqdm(pool.imap(process_func, lines_f, chunksize=chunksize), total=total_lines, desc='Processing Corpus'):
                            if words:
                                buffer.append(' '.join(words) + '\n')
                            if len(buffer) >= buffer_size:
                                cache_f.writelines(buffer)
                                buffer = []
                                
                            # 定期监控内存使用和垃圾回收
                            processed_lines += 1
                            if processed_lines % check_interval == 0:
                                mem_percent = psutil.virtual_memory().percent
                                if mem_percent > memory_threshold * 100:
                                    print(f"Memory Usage {mem_percent}% Exceeds Threshold {memory_threshold*100}%, Triggering Garbage Collection")
                                    gc.collect()
                                    print(f"Memory After GC: {psutil.virtual_memory().percent}%")



                    except Exception as e:
                        print(f"Error During Corpus Processing: {e}")
                        raise
                    finally:
                        if buffer:
                            cache_f.writelines(buffer)
                        del buffer
                        gc.collect()
                        
                        
    
    # 读取预处理后的语料
    print(f"Reading Preprocessed Corpus from {cache_corpus_txt_file}")
    try:
        sentences = LineSentence(cache_corpus_txt_file)
        
        # 添加短语检测
        phrases = Phrases(sentences, min_count=min_count, threshold=15.0, delimiter='_')
        sentences = phrases[sentences]
        
        # 替换下划线
        if lang == 'chinese':
            sentences = [[word.replace('_', '') for word in sentence] for sentence in sentences]
        elif lang == 'english':
            sentences = [[word.replace('_', ' ') for word in sentence] for sentence in sentences]
        else:
            raise ValueError(f"Unsupported Language: {lang}")
        
    except Exception as e:
        print(f"Error Reading Preprocessed Corpus: {e}")
        del e
        gc.collect()

    # 训练模型
    print(f"Start Training Word2Vec")
    
    model = word2vec.Word2Vec(
            sentences=sentences,
            vector_size=vector_size,
            window=window_size,
            min_count=min_count,
            workers=max_worker,  # 使用所有CPU核心
            epochs=max_iter,
            **kwargs)
    
    del sentences
    gc.collect()
    
    
    # 保存模型
    bin_model_name = f'{corpus_name}-Word2Vec.{vector_size}.{window_size}.bin'
    txt_model_name = f'{corpus_name}-Word2Vec.{vector_size}.{window_size}.txt'
    bin_file = output_dir / bin_model_name
    txt_file = output_dir / txt_model_name
    if not only_binary:
        model.wv.save_word2vec_format(str(bin_file), binary=True)
        model.wv.save_word2vec_format(str(txt_file), binary=False)
        end = time.time()
        duration = int(end - start)
        print(f'Word2Vec Training Cost {duration} s. \nOutput Saved To: {txt_file}')
        print(f'Output Saved To: {bin_file}')
    else:
        model.wv.save_word2vec_format(str(bin_file), binary=True)
        end = time.time()
        duration = int(end - start)
        print(f'Word2Vec Training Cost {duration} s. \nOutput Saved To: {bin_file}')


    
    result = model.wv
    del model
    return result