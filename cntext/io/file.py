import pandas as pd
import glob
import pathlib
from tqdm import tqdm
import fitz
import docx
from pathlib import Path
import concurrent.futures
import os

def read_pdf(file):
    """
    读取PDF文件，返回文本内容
    """
    with fitz.open(file) as doc:
        return "".join((p.get_text()+ '\n' for p in doc))



def read_docx(file):
    """
    读取docx文件，并返回其中的文本内容
    """
    doc = docx.Document(file)
    return "".join((p.text+ '\n' for p in doc.paragraphs))



######################################################################################################################



def read_file(file, encoding='utf-8', **kwargs):
    """
    读取文件数据, 支持 .txt, .csv, .pdf,.docx, .json, .dta, .feather, .parquet, etc.
    """
    file_path = Path(file)
    fext = file_path.suffix.lower()

    try:
        if fext == '.txt':
            with file_path.open('r', encoding=encoding, **kwargs) as f:
                text = f.read()
        elif fext == '.docx':
            text = read_docx(file_path)
        elif fext == '.pdf':
            text = read_pdf(file_path, **kwargs)
        elif fext == '.xls':
            text = pd.read_excel(file_path, **kwargs)
        elif fext == '.csv':
            text = pd.read_csv(file_path, encoding=encoding, **kwargs)
        elif fext == '.dta':
            text = pd.read_stata(file_path, **kwargs)
        elif fext == '.json':
            text = pd.read_json(file_path, encoding=encoding, **kwargs)
        elif fext == '.feather':
            text = pd.read_feather(file_path, **kwargs)
        elif fext == '.parquet':
            text = pd.read_parquet(file_path, **kwargs)
        else:
            print(f'不支持的文件格式: {fext}')
            text = pd.DataFrame(dict())
            
        if not isinstance(text, pd.DataFrame):
            df = pd.DataFrame({
                'doc': text,
                'file': str(file_path)
            }, index=[0])
        else:
            df = text
        return df

    except Exception as e:
        print(f'读取文件 {file_path} 时出错: {str(e)}')
        return pd.DataFrame()





def read_files(fformat='*.*', encoding='utf-8', recursive=True, **kwargs):
    """
    批量读取符合fformat格式的所有文件数据，返回DataFrame(含doc和file两个字段)。
    
    Args:
        fformat (str):  fformat格式支持 txt/pdf/docx/xlsx/csv等。 "*"表示通配符；"*.txt"匹配当前代码所在路径内的所有txt
        recursive (bool, optional): 是否搜寻某路径下所有层级的文件夹内的文件. 默认True
        
    Returns:
        DataFrame
    """
    
    files = get_files(fformat=fformat, recursive=recursive)
    dfs = []
    
    # 根据CPU核数设置线程数，采用CPU核数*2的线程池
    max_workers = min((os.cpu_count() or 1) * 2, 64)
    
    # 使用ProcessPoolExecutor代替ThreadPoolExecutor，提高CPU密集型任务效率
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for file in files:
            futures.append(executor.submit(read_file, file, encoding=encoding, **kwargs))
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Reading files..."):
            try:
                dfs.append(future.result())
            except Exception as e:
                print(f"Error reading file: {e}")
    
    all_df = pd.concat(dfs, axis=0, ignore_index=True)
    return all_df




def get_files(fformat='*.txt', recursive=True):
    """
    查看符合fformat路径规则的所有的文件

    Args:
        fformat (str):  fformat格式支持 txt/pdf/docx/xlsx/csv等。 "*"表示通配符；"*.txt"匹配当前代码所在路径内的所有txt
        recursive (bool, optional): 是否搜寻某路径下所有层级的文件夹内的文件. 默认True

    Returns:
        文件路径列表
    """
    file_list = glob.glob(fformat, recursive=recursive)
    
    #unify the sep
    file_list = [file_path.replace('\\', '/') for file_path in file_list]
    return file_list