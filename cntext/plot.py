from nltk.tokenize import word_tokenize
import jieba
import os
import logging
jieba_logger = logging.getLogger('jieba')
jieba_logger.setLevel(logging.CRITICAL)











def matplotlib_chinese():
    """
    支持中文matplotlib可视化

    Returns:
        返回plt
    """
    
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from distutils.version import LooseVersion
    import matplotlib_inline
    matplotlib_inline.backend_inline.set_matplotlib_formats('png', 'svg')
    import scienceplots
    #import platform
    import matplotlib
    plt.style.use(['nature', 'no-latex', 'cjk-sc-font'])
    #system = platform.system()
    
    font_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'font'))
    font_files = font_manager.findSystemFonts(fontpaths=[font_path])
    is_support_createFontList = LooseVersion(matplotlib.__version__) < '3.2'
    if is_support_createFontList:
        font_list = font_manager.createFontList(font_files)
        font_manager.fontManager.ttflist.extend(font_list)
    else:
        for fpath in font_files:
            font_manager.fontManager.addfont(fpath)
    matplotlib.rc('font', family='Source Han Sans CN')
    return plt
    

    #font = {
    #    'family': 'SimHei' if system == 'Windows' else 'Arial Unicode MS' if system == 'Darwin' else #'sans-serif'
    #}
    #matplotlib.rc('font', **font)
    #return plt
    




def lexical_dispersion_plot1(text, targets_dict, lang='chinese', figsize=(8, 5), title="词汇离散图", prop=True, dpi=300):
    """
    词汇分散图可视化， 对某一个文本text， 可视化不同目标类别词targets_dict在文本中出现位置

    Args:
        text (str): 文本数据
        targets_dict (dict): 目标类别词字典； targets_dict={'pos': ['开心', '快乐'], 'neg': ['悲伤', '难过']}
        lang (str, optional): 文本text的语言类型，默认'chinese'.
        figsize (tuple, optional): 图的长宽尺寸. 默认 (8, 5).
        title (str, optional): 图的标题；
        prop (bool, optional): 横坐标字符位置是否为相对位置. 默认True，横坐标索引值取值范围0 ~ 100
        dpi(int, optional): 设置图片保存时的清晰度

    Returns:
        _type_: _description_
    """
    plt = matplotlib_chinese()
    category_positions = {category: [] for category in targets_dict}

    words = jieba.lcut(text) if lang=='chinese' else word_tokenize(text.lower())


    for x, word in enumerate(words):
        for category, targets in targets_dict.items():
            if word in targets:
                category_positions[category].append(x * (100 if prop else 1) / len(words) if prop else x)
                break
    
    _, ax = plt.subplots(figsize=figsize)
    ax.yaxis.set_ticks_position('none')
    
    for idx, (category, positions) in enumerate(category_positions.items()):
        if positions:
            ax.plot(positions, [idx]*len(positions), '|', label=category)
    
    ax.set_yticks(list(range(len(category_positions))), list(category_positions.keys()))
    ax.set_ylim(-1, len(category_positions))
    ax.set_title(title)
    
    # 根据 prop 调整横坐标标签
    ax.set_xlabel("文本相对位置 (%)" if prop else "文本绝对位置")
    
    
    # 调整横坐标范围和刻度，仅当 prop=True 时生效
    if prop:
        ax.set_xlim(0, 100)
        ax.xaxis.set_ticks(range(0, 101, 10))  # 设置每10%一个刻度标记
    plt.savefig('plot.png', dpi=dpi, bbox_inches='tight')  # dpi设置分辨率，bbox_inches去除多余空白
    return ax



def lexical_dispersion_plot2(texts_dict, targets, lang='chinese', figsize=(12, 6), title="特定词汇在不同文本来源的相对离散图", dpi=300):
    """
    词汇分散图可视化， 对某几个文本texts_dict， 可视化某些目标词targets在文本中出现相对位置(0~100)

    Args:
        texts_dict (dict): 多个文本的字典数据。形如{'source1': 'source1的文本内容', 'source2': 'source2的文本内容'}
        targets (list): 目标词列表
        lang (str, optional): 文本数据texts_dict的语言类型，默认'chinese'.
        figsize (tuple, optional): 图的长宽尺寸. 默认 (8, 5).
        title (str, optional): 图的标题；
        dpi(int, optional): 设置图片保存时的清晰度

    Returns:
        _type_: _description_
    """
    plt = matplotlib_chinese()
    if lang=='chinese':
        words_dict = {source: jieba.lcut(text) 
                      for source, text in texts_dict.items()}
    else:
        words_dict = {source: word_tokenize(text.lower()) 
                      for source, text in texts_dict.items()}

    # 存储每个文本来源中目标词的相对位置信息
    source_relative_positions = {source: [] for source in words_dict}
    
    for source, words in words_dict.items():
        text_length = len(words)
        for x, word in enumerate(words):
            if word in targets:
                relative_pos = x / text_length * 100  # 转换为百分比形式的相对位置
                source_relative_positions[source].append(relative_pos)
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.yaxis.set_ticks_position('none')
    
    # 绘制每个文本来源的目标词相对位置
    for idx, (source, positions) in enumerate(source_relative_positions.items()):
        if positions:
            ax.plot(positions, [idx]*len(positions), '|', label=source)
    
    # 设置y轴标签和范围
    ax.set_yticks(list(range(len(source_relative_positions))), list(source_relative_positions.keys()))
    ax.set_ylim(-1, len(source_relative_positions))
    ax.set_title(title)
    
    # 设置x轴标签为百分比形式的相对位置，并调整刻度
    ax.set_xlabel("文本内相对位置 (%)")
    ax.set_xlim(0, 100)  # 调整x轴范围到0-100%
    ax.xaxis.set_ticks(range(0, 101, 10))  # 每10%设置一个刻度标记
    plt.savefig('plot.png', dpi=dpi, bbox_inches='tight')  # dpi设置分辨率，bbox_inches去除多余空白
    #ax.legend()
    return ax

