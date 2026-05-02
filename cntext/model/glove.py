from pathlib import Path
import time
import subprocess
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec as glove2wv
import os
from ..model.utils import preprocess_line, get_optimal_threshold
import platform
import sys
import gc
from tqdm import tqdm
import psutil
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


def GloVe(corpus_file, lang='chinese', dict_file=None, stopwords_file=None, vector_size=100, window_size=15, min_count=5, max_memory=4.0, max_iter=15, x_max=10, chunksize=100000, only_binary=True):
    """
    使用Stanford GloVe代码工具训练GloVe模型。该函数会自动处理文本预处理(分词、去停词)、内存管理、参数调整等问题，确保训练过程顺利进行。
        
    Args:
        corpus_file (str): 输入语料文件路径（文本格式）。该文件为分词后的语料文件。utf-8编码。
        lang (str, optional): 语料文件的语言类型，默认为 'chinese'。
        dict_file (str, optional): 用户词典文件路径，默认为 None。utf-8编码。
        stopwords_file (str, optional): 停用词文件路径，默认为 None。utf-8编码。
        vector_size (int): 词向量维度，默认 100。
        window_size (int): 上下文窗口大小，默认 15。
        min_count (int): 忽略出现次数低于此值的单词，默认 5。
        max_worker (int): 使用的线程数，默认 8; 该参数越大速度越快。
        max_memory (int): 可供使用的最大内存大小，默认 4.0 ，单位为GB；  该参数越大，训练越快。
        max_iter (int): 训练的最大迭代次数，默认 15。
        x_max (int): 共现矩阵中元素的最大计数值，默认 10。
        chunksize (int, optional): 每次读取的行数。默认为100000。越大速度越快。
        only_binary (bool, optional): 是否只保存模型为二进制文件， 默认为True，只保存bin。 False时保存为txt和bin。
  
        
    Returns: 训练好的 GloVe 模型。
    """
    max_worker = int(cpu_count() * 0.8)
    memory_threshold = get_optimal_threshold()
    system = platform.system()
    if system == 'Windows':
        # https://github.com/hfxunlp/GloVe-win
        if sys.maxsize > 2**32:
            build_dir = Path(__file__).parent /  "build" / "glove-win_devc_x64"
        else:
            build_dir = Path(__file__).parent /  "build" / "glove-win_devc_x86"
        required_files = ["vocab_count.exe", "cooccur.exe", "shuffle.exe", "glove.exe"]
    else:
        build_dir = Path(__file__).parent /  "build" / "glove-unix"
        required_files = ["vocab_count", "cooccur", "shuffle", "glove"]
        
    
    # 检查必要的文件是否存在
    for file in required_files:
        if not (build_dir / file).exists():
            raise FileNotFoundError(f"Missing required file: {file} in {build_dir}")
        

    binary=2
    
    verbose=0
        
    start = time.time()
    
    output_dir = Path('output')
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_name = Path(corpus_file).stem  # 去除文件扩展名
    vocab_file = output_dir / f"{corpus_name}_GloVe_Vocab.txt"
    cooccurrence_file = output_dir / f"{corpus_name}_GloVe_Cooccurrence.bin"
    cooccurrence_shuf_file = output_dir / f"{corpus_name}_GloVe_Cooccurrence.shuf.bin"
    save_file = output_dir / f"{corpus_name}-vectors"
    cache_corpus_txt_file = output_dir / f"{corpus_name}_cache.txt"
    
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
                
                
    # 切换到原始工作目录
    original_dir = os.getcwd()
    
    
    

    try:
        print(f"Start Training GloVe")
        # Step 1: 构建词汇表
        vocab_process = subprocess.run(
            [str(build_dir / "vocab_count"),
             "-min-count", str(min_count),
             "-verbose", str(verbose)],
            stdin=open(cache_corpus_txt_file, "r", encoding='utf-8'),
            stdout=open(vocab_file, "w", encoding='utf-8'),
            check=True)
        del vocab_process

        # Step 2: 构建共现矩阵
        cooccur_process = subprocess.run(
            [str(build_dir / "cooccur"),
             "-memory", str(max_memory),
             "-vocab-file", str(vocab_file),
             "-verbose", str(verbose),
             "-window-size", str(window_size)],
            stdin=open(cache_corpus_txt_file, "r", encoding='utf-8'),
            stdout=open(cooccurrence_file, "wb"),
            check=True)
        del cooccur_process

        # Step 3: 打乱共现矩阵
        shuffle_process = subprocess.run(
            [str(build_dir / "shuffle"),
             "-memory", str(max_memory),
             "-verbose", str(verbose)],
            stdin=open(cooccurrence_file, "rb"),
            stdout=open(cooccurrence_shuf_file, "wb"),
            check=True)
        del shuffle_process

        # Step 4: 训练 GloVe 模型
        glove_process = subprocess.run(
            [str(build_dir / "glove"),
             "-save-file", str(save_file),
             "-threads", str(max_worker),
             "-input-file", str(cooccurrence_shuf_file),
             "-x-max", str(x_max),
             "-iter", str(max_iter),
             "-vector-size", str(vector_size),
             "-binary", str(binary),
             "-vocab-file", str(vocab_file),
             "-verbose", str(verbose)],
            check=True)
        del glove_process
            
            
        #save_file
        to_delete_txt = output_dir / f"{corpus_name}-vectors.txt"
        to_delete_bin = output_dir / f"{corpus_name}-vectors.bin"
        final_txt_file = output_dir / f"{corpus_name}-GloVe.{vector_size}.{window_size}.txt"
        final_bin_file = output_dir / f"{corpus_name}-GloVe.{vector_size}.{window_size}.bin"
        glove2wv(to_delete_txt, final_txt_file)
        os.remove(to_delete_txt)
        os.remove(to_delete_bin)
            
        os.remove(vocab_file)
        os.remove(cooccurrence_file)
        os.remove(cooccurrence_shuf_file)

            
        wv = KeyedVectors.load_word2vec_format(final_txt_file, binary=False)
        wv.save_word2vec_format(final_bin_file, binary=True)
        if only_binary:
            os.remove(final_txt_file)
            end = time.time()
            duration = int(end - start)
            print(f"\nGloVe Training Cost {duration} s. \nOutput Saved To: {final_bin_file}")
        else:
            end = time.time()
            duration = int(end - start)
            print(f"\nGloVe Training Cost {duration} s. \nOutput Saved To: {final_txt_file}")
            print(f"Output Saved To: {final_bin_file}")
        
        result = wv
        del wv
        gc.collect()
        return result
        
        
    except subprocess.CalledProcessError as e:
        print(f"Error Occurred During GloVe Training: {e}")
        del e
        gc.collect()
    finally:
        # 恢复原始工作目录
        os.chdir(original_dir)
    