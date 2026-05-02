import os
import pathlib
import yaml




def build_yaml_dict(data, yfile):
    """
    构建自己的yaml词典文件

    Args:
        data (dict): dict类型数据, 格式参考cntext内置的yaml文件
        yfile (str): yaml文件路径
    """
    #assert set(data.keys())==set(['Name', 'Desc', 'Refer', 'Category', 'Dictionary']), "The keys of the dictionary are not correct. Please check carefully!."
    #assert set(data.get('Dictionary').keys()) == set(data.get('Category')), "The categories of the dictionary and the categories list are not consistent."
    
    with open(yfile, 'w', encoding='utf-8') as file:
        yaml.dump(data, file, allow_unicode=True, indent=4, sort_keys=False)
        




def read_yaml_dict(yfile, is_builtin=True):
    """
    读取内置yaml词典，返回字典数据
    
    Usage:
        >>>dictdata = read_dict_yml(yfile='test.yaml')
        {'Name': 'Sentiment Dictionray',
         'Desc': 'This dictionary contains two lists of emotion words: positive and negative.',
         'Refer': 'Reference article',
         'Category': ['positive', 'negative'],
         'Dictionary': {
             'negative': ['sad', 'bad', 'error'],
             'positive': ['happy', 'bright', 'successfull']
             }
         }
    
    Args:
        yfile (str): 字典yaml文件路径； 当 is_builtin=True, 该函数读取的是cntext的内置yaml词典文件.
        is_builtin (bool, optional):  是否默认读取cntext内置的yaml词典文件；默认读取内置词典.

    Returns:
        dict data, format referer to the Usage above
    """
    if is_builtin==True:
        pathchain = ['data', yfile]
        yfile = pathlib.Path(__file__).parent.joinpath(*pathchain)

    with open(yfile, 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)
    return data




def get_dict_list():
    """
    获取cntext内置的yaml词典文件列表
    
    Returns:
        返回yaml词典文件列表
    """
    #data文件夹内存放着.yaml文件
    dict_file_path = pathlib.Path(__file__).parent.joinpath('data')
    dicts = [f for f in os.listdir(dict_file_path) if '.yaml' in f]
    return dicts

