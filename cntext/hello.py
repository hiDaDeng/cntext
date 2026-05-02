# cntext/hello.py

from IPython.display import display, HTML
import sys

_called = False



def _create_welcome_html():
    return """
    <div style="
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif;
        max-width: 700px;
        margin: 20px auto;
        padding: 20px;
        border-radius: 14px;
        background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
        color: white;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.18);
        border: 1px solid rgba(255, 255, 255, 0.1);
        line-height: 1.6;
    ">
        <!-- 标题区 -->
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
            <span style="font-size: 1.6em;">🎉</span>
            <h1 style="margin: 0; font-size: 1.6em; font-weight: 600;">
                欢迎使用 <strong style='font-weight: 700;'>cntext</strong>
            </h1>
        </div>
        <p style="margin: 0 0 16px 0; opacity: 0.9; font-size: 1em;">
            中文文本分析工具包 —— 让文本理解更简单
        </p>

        <!-- 文档链接 -->
        <div style="
            background: rgba(255,255,255,0.1);
            border-radius: 8px;
            padding: 10px 14px;
            margin: 16px 0;
            font-size: 0.95em;
        ">
            <strong>📖 官方文档</strong>
            <a href='https://textdata.cn/' target='_blank'
               style='color: #a29bfe; text-decoration: none; margin-left: 6px;'>
               https://textdata.cn/
            </a>
        </div>

        <!-- 五大模块 -->
        <details style="margin: 16px 0;" open>
            <summary style="
                font-weight: 600;
                font-size: 1.05em;
                color: white;
                margin-bottom: 8px;
                cursor: pointer;
            ">
                🛠️ 五大核心模块
            </summary>
            <div style="
                margin-top: 10px;
                display: grid;
                grid-template-columns: 170px 1fr;
                gap: 8px;
                font-size: 0.95em;
                color: white;
            ">
                <div><strong>io</strong></div>
                <div>读取/清洗文本（PDF, DOCX, 编码修复）</div>

                <div><strong>model</strong></div>
                <div>词向量训练（Word2Vec/GloVe）与评估</div>

                <div><strong>stats</strong></div>
                <div>词频/情感/可读性/EPU/相似度</div>

                <div><strong>mind</strong></div>
                <div>语义分析（概念轴/语义投影）</div>

                <div><strong>llm</strong></div>
                <div>大模型文本分析（新）</div>
            </div>
        </details>

        <!-- 常用函数 -->
        <details style="margin: 16px 0;" open>
            <summary style="
                font-weight: 600;
                font-size: 1.05em;
                color: white;
                margin-bottom: 8px;
                cursor: pointer;
            ">
                💡 常用函数
            </summary>
            <div style="
                margin-top: 10px;
                display: grid;
                grid-template-columns: 170px 1fr;  /* 增加第一列宽度 */
                gap: 12px;                        /* 增加列间距，防止重叠 */
                font-family: monospace;
                font-size: 0.9em;
                color: white;
            ">
                <div>ct.read_files(...)</div>
                <div>批量读文件</div>

                <div>ct.word_count(...)</div>
                <div>词频统计</div>

                <div>ct.sentiment(...)</div>
                <div>情感分析</div>

                <div>ct.GloVe(...)</div>
                <div>训练GloVe</div>

                <div>ct.llm(...)</div>
                <div>大模型（新）</div>
            </div>
        </details>

        <!-- 底部提示 -->
        <p style="
            text-align: right;
            font-size: 0.9em;
            margin: 20px 0 0 0;
            opacity: 0.7;
            font-style: italic;
        ">
            ✨ 输入 <code style="background: rgba(255,255,255,0.2); padding: 2px 4px; border-radius: 3px;">ct.hello()</code> 可再次查看此页面
        </p>
    </div>
    """


def hello():
    global _called

    if 'ipykernel' not in sys.modules:
        print("💡 提示：此功能在 Jupyter Notebook 中有图形化展示效果。")
        return

    display(HTML(_create_welcome_html()))
    _called = True


def welcome():
    """hello() 的同义词，提供语义别名"""
    hello()


# 可选：定义 __all__ 控制导入行为
__all__ = ['hello', 'welcome']