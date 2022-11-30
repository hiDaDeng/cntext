from setuptools import setup
import setuptools

setup(
    name='cntext',     # 包名字
    version='1.8.2',   # 包版本
    description='Chinese text analysis library, which can perform word frequency statistics, dictionary expansion, sentiment analysis, similarity, readability, co-occurrence analysis, social calculation (attitude, prejudice, culture) on texts',   # 简单描述
    author='大邓',  # 作者
    author_email='thunderhit@qq.com',  # 邮箱
    url='https://github.com/hidadeng/cntext',      # 包的主页
    packages=setuptools.find_packages(),
    package_data = {'':['files/*.pkl']}, 
    #install_requires=['jieba', 'numpy', 'mittens', 'scikit-learn==1.0', 'numpy==1.20.0', 'matplotlib', 'pyecharts', 'gensim==4.0.0', 'nltk'],
    install_requires=['jieba', 'numpy', 'mittens', 'scikit-learn', 'numpy', 'matplotlib', 'pyecharts', 'gensim', 'nltk'],
    python_requires='>=3.5',
    license="MIT",
    keywords=['chinese', 'text mining', 'sentiment', 'sentiment analysis', 'natural language processing', 'sentiment dictionary development', 'text similarity'],
    long_description=open('README.md', encoding='utf-8').read(), # 读取的Readme文档内容
    long_description_content_type="text/markdown")  # 指定包文档格式为markdown
    #py_modules = ['eventextraction.py']
