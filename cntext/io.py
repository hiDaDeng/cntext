from pdfdocx import read_pdf, read_docx
import pandas as pd
import glob
import os
import chardet



def detect_encoding(file, num_lines=100):
    """
    Detect encoding of file

    Args:
        file (str): file path
        num_lines (int, optional):  Defaults to 100.

    Returns:
        encoding type
    """
    with open(file, 'rb') as f:
        detector = chardet.UniversalDetector()
        for line in f:
            detector.feed(line)
            if detector.done:
                break
            num_lines -= 1
            if num_lines == 0:
                break
    detector.close()
    return detector.result['encoding']



def read_file(file, encoding='utf-8', **kwargs):
    """
    Read data from common format file, support .txt, .csv, .pdf, .docx, .json, .dta, etc.

    Args:
        file (str): file path
        encoding (str, optional): Defaults to 'utf-8'.
        **kwargs: other arguments for pd.read_csv, pd.read_excel, pd.read_stata, pd.read_json, pdfdocx.read_pdf, pdfdocx.read_docx
        
    Returns:
        DataFrame
    """
    if '.txt' in file or '.TXT' in file:
        text = open(file, 'r', encoding=encoding, **kwargs).read()
    elif '.docx' in file or '.DOCX' in file:
        text = read_docx(file)
    elif '.pdf' in file or '.PDF' in file:
        text = read_pdf(file, **kwargs)
    elif '.xls' in file or '.xlsx' in file:
        text = pd.read_excel(file,**kwargs)
    elif '.csv' in file or '.CSV' in file:
        text = pd.read_csv(file, encoding=encoding, **kwargs)
    elif '.dta' in file or '.DTA' in file:
        text = pd.read_stata(file, **kwargs)
    elif '.json' in file or '.JSON' in file:
        text = pd.read_json(file, encoding=encoding, **kwargs)
    else:
        print('无能为力')
        text = pd.DataFrame(dict())
        
    if type(text)!=pd.DataFrame:
        df = pd.DataFrame({
            'doc': text,
            'file': file
        }, index=[0])
    else:
        df = text
    
    return df



def get_files(fformat='*.txt', recursive=True):
    """
    Get a list of file path in a folder

    Args:
        fformat (str): filter files, the default value is '*.txt', which means this function only returns a file path list of TXT files. Defaults to '*.txt'. Other options are '*.pdf', '*.docx', '*.csv', '*.xls', '*.xlsx', '*.txt' .If you dont know the subdirectory of the folder, you can use '**/*.txt',
        '**/*.pdf', '**/*.docx', '**/*.csv', '**/*.xls', '**/*.xlsx', '**/*.txt' 
        recursive (bool, optional): Whether to recursive search in folder. Defaults to True.

    Returns:
        a list of file  path
    """
    file_list = glob.glob(fformat, recursive=recursive)
    
    #unify the sep
    file_list = [file_path.replace('\\', '/') for file_path in file_list]
    return file_list



def read_files(fformat='*.*', encoding='utf-8', recursive=True, **kwargs):
    """
    Read files from specificed folder path. 
    
    Args:
        fformat (str): filter files, the default value is '*.txt', which means this function only returns a file path list of TXT files. Defaults to '*.txt'. Other options are '*.pdf', '*.docx', '*.csv', '*.xls', '*.xlsx', '*.txt' , '*.dta', '*.json'  .If you dont know the subdirectory of the folder, you can use '*/*.txt',
        '*/*.pdf', '*/*.docx', '*/*.csv', '*/*.xls', '*/*.xlsx', '*/*.txt' , '*/*.dta' , '*/*.json' 
    
        recursive (bool, optional): Whether to recursive search in folder. Defaults to True.

    Returns:
        DataFrame
    """
    dfs = []
    files = get_files(fformat=fformat, recursive=recursive)
    for file in files:
        try:
            dfs.append(read_file(file, encoding=encoding, **kwargs))
        except:
            pass
    all_df = pd.concat(dfs, axis=0, ignore_index=True)
    return all_df