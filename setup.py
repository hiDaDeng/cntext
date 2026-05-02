from setuptools import setup
import setuptools

setup(
    name='cntext',     # 包名字
    version='2.2.1',   # 包版本
    description='Chinese text analysis library, which can perform word frequency statistics, dictionary expansion, sentiment analysis, similarity, readability, co-occurrence analysis, social computing (attitude, prejudice, culture) on texts',   # 简单描述
    author='大邓',  # 作者
    author_email='thunderhit@qq.com',  # 邮箱
    url='https://github.com/hidadeng/cntext',      # 包的主页
    packages=setuptools.find_packages(),
    package_data = {'':['dict/*.yaml', 'font/*.ttf']}, 
    include_package_data=True,
    install_requires=['jieba>=0.42', 'scikit-learn==1.5.0', 'numpy==1.26.4', 'matplotlib',
                      'gensim==4.3.2', 'nltk', 'pandas', 'chardet', 'h5py', 'networkx', 'distinctiveness',
                      'tqdm', 'python-docx', 'scipy>=1.12.0', 'scienceplots', 'PyMuPDF', 'ftfy', 'opencc-python-reimplemented',
                      "ollama>=0.2.1", "pydantic>=2.8.2", "instructor>=1.6.0", "openai>=1.61.1","contractions>=0.1.73"],
    python_requires='>=3.5',
    license="MIT",
    keywords=['chinese', 'text mining', 'sentiment', 'sentiment analysis', 'natural language processing', 'sentiment dictionary development', 'text similarity', 'GloVe', 'word2vec'],
    long_description=open('README.md', encoding='utf-8').read(), # 读取的Readme文档内容
    long_description_content_type="text/markdown")  # 指定包文档格式为markdown
    #py_modules = ['eventextraction.py']
