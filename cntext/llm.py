import asyncio
import nest_asyncio
import pandas as pd
from typing import Union, List, Dict, Any, Optional
from openai import AsyncOpenAI
import instructor
from pydantic import create_model
from aiolimiter import AsyncLimiter
import warnings

# 应用 nest_asyncio，解决 Jupyter 中 event loop 已运行的问题
nest_asyncio.apply()


# ======================
# 工具函数：安静打印
# ======================

def _is_notebook():
    try:
        shell = get_ipython().__class__.__name__
        return shell in ['ZMQInteractiveShell', 'TerminalInteractiveShell']
    except:
        return False

_printed_messages = set()

def _quiet_print(msg: str, verbose: bool = True, once: bool = True):
    if not verbose:
        return
    if once and msg in _printed_messages:
        return
    _printed_messages.add(msg)

    if _is_notebook():
        from IPython.display import display, HTML
        html_msg = f"<small style='color: #555;'>[cntext2x] {msg}</small>"
        display(HTML(html_msg))
    else:
        print(f"[cntext2x] {msg}")


# ======================
# 异步批量处理函数（简化版）
# ======================

async def _llm_async_batch(
    inputs: List[str],  # 只接受字符串列表
    task: str,
    prompt: Optional[str],
    output_format: Optional[Dict[str, Any]],
    base_url: str,
    api_key: str,
    model_name: str,
    temperature: float,
    max_retries: int,
    rate_limit: Optional[Union[int, float]] = None,  # 新：统一限速参数
):
    """内部异步批量处理函数（仅支持字符串输入）"""
    

    limiter = None

    if rate_limit is not None:
        if isinstance(rate_limit, int):
            # 整数：每分钟请求数（适合 API 文档如 100 次/分钟）
            limiter = AsyncLimiter(rate_limit, 60)
        elif isinstance(rate_limit, (float, int)):
            # 浮点数：每秒请求数（如 5.0 表示 5 QPS）
            limiter = AsyncLimiter(rate_limit, 1)
    
    # 加载任务配置
    if prompt is None or output_format is None:
        if task not in TASKS:
            available = ", ".join(TASKS.keys())
            raise ValueError(f"不支持的任务: {task}，可用任务: {available}")
        config = TASKS[task]
        prompt = prompt or config["prompt"]
        output_format = output_format or config["output_format"]

    # 创建异步客户端
    aclient = instructor.from_openai(
        AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=30),
        mode=instructor.Mode.MD_JSON,
    )

    # 构建 Pydantic 模型
    type_map = {
        'str': str, 'int': int, 'float': float,
        'bool': bool, 'list[str]': List[str], List[str]: List[str]
    }
    fields = {}
    for k, v in output_format.items():
        typ = type_map.get(v.lower() if isinstance(v, str) else v, v)
        fields[k] = (typ, ...)
    ResponseModel = create_model('ResponseModel', **fields)

    # 单个请求协程（只处理字符串）
    async def _call(text: str):
        try:
            if limiter:
                await limiter.acquire()
            resp = await aclient.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": text}
                ],
                response_model=ResponseModel,
                temperature=temperature,
                max_retries=max_retries,
            )
            result = resp.model_dump()
        except Exception as e:
            return {"error": str(e), "text": text}
        return result

    # 并发执行
    tasks = [_call(text) for text in inputs]
    return await asyncio.gather(*tasks, return_exceptions=False)


# ======================
# 主函数 llm（仅支持 str 或 List[str]）
# ======================

def llm(
    text: Union[str, List[str]],  # 明确只支持 str 和 List[str]
    task: str = "sentiment",
    prompt: Optional[str] = None,
    output_format: Optional[Dict[str, Any]] = None,
    backend: Optional[str] = "ollama",
    base_url: Optional[str] = None,
    api_key: str = "",
    model_name: str = "qwen2.5:3b",
    temperature: float = 0,
    max_retries: int = 3,
    rate_limit: Optional[Union[int, float]] = None,  # 如 100（次/分钟）或 5.0（QPS）
    return_df: bool = False,
    verbose: bool = True,
) -> Union[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame]:
    """
    调用大模型执行结构化文本分析任务（如情感分析、关键词提取、分类等）。

    支持：
    - 本地模型：Ollama (11434), LM Studio (1234)
    - 远程服务：阿里云、百度千帆、自建API等（通过 base_url）

    Args:
        text (str): 待分析的文本内容
        task (str): 预设任务名称，默认为 'sentiment'。可用任务见 TASKS.keys()
        backend (str, optional): 快捷后端别名：
            - 'ollama' → http://127.0.0.1:11434/v1
            - 'lmstudio' 或 'lms' → http://localhost:1234/v1
            - None → 需配合 base_url 使用
        base_url (str, optional): 自定义模型服务地址，优先级高于 backend
            示例：
            - 远程：https://dashscope.aliyuncs.com/compatible-mode/v1
            - 内网：http://192.168.1.10:11434/v1
            - 本地：http://localhost:1234/v1
        api_key (str): API 密钥，远程服务必填，本地通常为 "EMPTY"
        model_name (str): 模型名称（需服务端已加载）
        temperature (float): 生成温度，0 表示确定性输出
        max_retries (int): 失败重试次数
        rate_limit (int or float, optional): 限速
            - int: 每分钟最多请求数，如 100
            - float: 每秒最多请求数（QPS），如 5.0
        return_df (bool): 是否返回 DataFrame
        verbose (bool): 是否输出连接信息
        prompt (str, optional): 自定义系统提示语
        output_format (dict, optional): 自定义输出结构，如 {'label': str, 'score': float}

    Returns:
        dict or pd.DataFrame: 结构化结果，如 {'label': 'pos', 'score': 0.95}

    Example:
        # 本地 Ollama
        llm("服务很棒", backend="ollama", model_name="qwen2.5:3b")

        # 阿里云通义千问
        llm("服务很棒",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key="sk-xxx",
            model_name="qwen-plus")

        # 自定义任务
        llm("总结这段话",
            prompt="请生成一句话摘要",
            output_format={"summary": str},
            base_url="http://127.0.0.1:11434/v1")
    """
    # ====== 1. 确定 base_url ======
    final_base_url = base_url
    if final_base_url is None:
        if backend == "ollama":
            final_base_url = "http://127.0.0.1:11434/v1"
        elif backend in ("lmstudio", "lms"):
            final_base_url = "http://localhost:1234/v1"
        else:
            raise ValueError(f"不支持的 backend: {backend}。请提供 base_url")
    else:
        if verbose:
            _quiet_print(f"🌐 使用自定义 base_url: {final_base_url}", once=True)

    if verbose:
        _quiet_print(f"✅ 连接模型服务: {final_base_url}", once=True)

    # ====== 2. 输入类型检查 ======
    if isinstance(text, str):
        is_single = True
        inputs = [text]
    elif isinstance(text, list):
        if not all(isinstance(t, str) for t in text):
            raise TypeError("当输入为列表时，所有元素必须是字符串")
        is_single = False
        inputs = text
    else:
        raise TypeError("text 必须是字符串或字符串列表")

    # ====== 3. 异步主函数 ======
    async def main():
        return await _llm_async_batch(
            inputs=inputs,
            task=task,
            prompt=prompt,
            output_format=output_format,
            base_url=final_base_url,
            api_key=api_key or "EMPTY",
            model_name=model_name,
            temperature=temperature,
            max_retries=max_retries,
            rate_limit=rate_limit
        )

    # ====== 4. 执行 ======
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = None

    if loop is None or not loop.is_running():
        results = asyncio.run(main())
    else:
        results = asyncio.run(main())  # nest_asyncio 已 apply

    # ====== 5. 返回格式处理 ======
    if return_df:
        return pd.DataFrame(results)
    return results[0] if is_single else results


# ======================
# 老函数：软弃用（兼容旧用户）
# ======================

def text_analysis_by_llm(*args, **kwargs):
    warnings.warn("已弃用，请使用 ct.llm()", DeprecationWarning, stacklevel=2)
    return llm(*args, **kwargs)

def analysis_by_llm(*args, **kwargs):
    warnings.warn("已弃用，请使用 ct.llm()", DeprecationWarning, stacklevel=2)
    return llm(*args, **kwargs)


# ======================
# 任务模板（保持不变）
# ======================







llm.tasks_list = lambda: list(TASKS.keys())
llm.tasks_get = lambda name: TASKS.get(name) or _raise(...)


# 或更优雅：封装为对象
class _LLMTasks:
    @staticmethod
    def list():
        return list(TASKS.keys())
    
    @staticmethod
    def get(name: str):
        if name not in TASKS:
            available = ", ".join(TASKS.keys())
            raise ValueError(f"不支持的任务: {name}，可用任务: {available}")
        return TASKS[name]
    
    @staticmethod
    def show(name: str):
        return _show_task(name)

llm.tasks = _LLMTasks()


#10 个专为中文场景设计的结构化任务模板
TASKS = {
    "sentiment": {
        "prompt": "分析评论的情感倾向：返回情感类别 label（pos 表示正面，neg 表示负面，neutral 表示中性）和情感分值 score（取值范围 -1~1，负数为负面）。结果返回JSON格式，格式为{'label': 'pos', 'score': 0.5}",
        "output_format": {
            "label": "str",
            "score": "float"
        }
    },
    "emotion": {
        "prompt": "识别文本中的主要情绪类型：从 [开心, 愤怒, 悲伤, 惊讶, 厌恶, 恐惧, 中性] 中选择最匹配的一项，返回情绪类型 emotion 和置信度 confidence（0~1）。结果返回JSON格式，格式为{'emotion': '开心', 'confidence': 0.8}",
        "output_format": {
            "emotion": "str",
            "confidence": "float"
        }
    },
    "classify": {
        "prompt": "将文本分类到最匹配的类别中：可选类别为 [科技, 体育, 娱乐, 财经, 教育, 医疗, 军事, 其他]，返回分类 category 和简要理由 reason。结果返回JSON格式，格式为{'category': '科技', 'reason': '文本中出现多次IT、AI等特征词，因此标记为科技。'}",
        "output_format": {
            "category": "str",
            "reason": "str"
        }
    },
    "intent": {
        "prompt": "识别用户的意图：从 [咨询, 投诉, 表扬, 购买, 建议, 其他] 中选择最匹配的一项，返回意图 intent 和置信度 confidence（0~1）。结果返回JSON格式，格式为{'intent': '咨询', 'confidence': 0.8'}",
        "output_format": {
            "intent": "str",
            "confidence": "float"
        }
    },
    "keywords": {
        "prompt": "提取文本中最相关的 3 个关键词，按重要性从高到低排序，返回关键词列表 keywords。结果返回JSON格式，格式为{'keywords': ['IT', '科技', 'AI']}",
        "output_format": {
            "keywords": "list[str]"
        }
    },
    "entities": {
        "prompt": "提取文本中的人名、地名、组织名，如果没有则返回空列表，返回 persons（人名）、locations（地名）、organizations（组织名）。结果返回JSON格式，格式为{'persons': ['张三', '李四'], 'locations': ['北京', '上海'], 'organizations': ['公司A', '公司B']}",
        "output_format": {
            "persons": "list[str]",
            "locations": "list[str]",
            "organizations": "list[str]"
        }
    },
    "summarize": {
        "prompt": "用一句话总结文本内容，不超过 30 个汉字，返回摘要 summary。结果返回JSON格式，格式为{'summary': '这是一个关于IT的文章，主要介绍了AI的发展。'}",
        "output_format": {
            "summary": "str"
        }
    },
    "rewrite": {
        "prompt": "用更简洁、流畅的方式重写该句，保持原意，返回改写后文本 rewritten。结果返回JSON格式，格式为{'rewritten': '这是修改后的内容: 。。。'}",
        "output_format": {
            "rewritten": "str"
        }
    },
    "quality": {
        "prompt": "对文本质量进行评分（0~1）：综合考虑逻辑性、表达清晰度和信息完整性，返回评分 score 和简要反馈 feedback。结果返回JSON格式，格式为{'score': 0.8, 'feedback': '文本逻辑性强，表达清晰，信息完整，给出0.8分。'}",
        "output_format": {
            "score": "float",
            "feedback": "str"
        }
    },
    "similarity": {
        "prompt": "判断两段文本的语义相似度（-1~1）：-1 表示完全相反，0 表示无关，1 表示几乎相同，返回相似度 similarity 和判断理由 reason。结果返回JSON格式，格式为{'similarity': 0.8, 'reason': '这两段文本的内容非常相似，都在讨论IT领域的发展。'}",
        "output_format": {
            "similarity": "float",
            "reason": "str"
        }
    }
}