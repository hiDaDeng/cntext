# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'cntext2.x'
copyright = '2025, 大邓'
author = '大邓'
release = '2.1.5'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# 添加扩展
extensions = [
    'myst_parser',
]

templates_path = ['_templates']
exclude_patterns = []

language = 'zh_CN'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'

# 添加静态文件支持
html_static_path = ['_static']

# 添加自定义JavaScript
#html_js_files = [
#    'clipboard.js',
#    'copybutton.js',
#]

# 支持的文件扩展名
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}
