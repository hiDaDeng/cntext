# 六、LLM模块


目前大模型本地化使用越来越方便，

| 模块      | 函数(类)                                                                                            | 功能                   |
| --------- | --------------------------------------------------------------------------------------------------- | ---------------------- |
| **_LLM_**   | **ct.llm(text, prompt, output_format, task, backend, base_url, api_key, model_name, temperature)** | 调用大模型执行结构化文本分析任务（如情感分析、关键词提取、分类等）。 |


### 6.1 ct.llm()

使用大模型（本地或 API）进行文本分析，从非结构化的文本数据中识别模式、提取关键信息、理解语义，并将其转化为结构化数据以便进一步分析和应用。

<br>

```python
ct.llm(text, prompt, output_format, task, backend, base_url, api_key, model_name, temperature)
```

- **text**: 待分析的文本内容
- **task**: 预设任务名称，默认为 'sentiment'。
- **prompt**: 自定义系统提示语
- **output_format**: 自定义输出结构，如 {'label': str, 'score': float}
- **backend**: 快捷后端别名：
            - 'ollama' → http://127.0.0.1:11434/v1
            - 'lmstudio' 或 'lms' → http://localhost:1234/v1
            - None → 需配合 base_url 使用
- **base_url**: 自定义模型服务地址，优先级高于 backend
            示例：
            - 远程：https://dashscope.aliyuncs.com/compatible-mode/v1
            - 内网：http://192.168.1.10:11434/v1
            - 本地：http://localhost:1234/v1
- **api_key**: API 密钥，远程服务必填，本地通常为 "EMPTY"
- **model_name**: 模型名称（需服务端已加载）
- **temperature**: 生成温度，0 表示确定性输出

<br>

**实验数据为外卖评论， 今天咱们做个有难度的文本分析任务，从不同维度(味道、速度、服务)对外卖评论进行打分(-1.0~1.0)**。

![](img/28-llm-analysis.png)<br>

```python
import cntext as ct

PROMPT = '从口味taste、速度speed、服务service三个维度， 对外卖评论内容进行文本分析， 分别返回不同维度的分值(分值范围-1.0 ~ 1.0)'
BASE_URL = 'https://dashscope.aliyuncs.com/compatible-mode/v1'
API_KEY = '你的API-KEY'
MODEL_NAME = 'qwen-max'

#味道、速度、服务
OUTPUT_FORMAT = {'taste': float, 'speed': float, 'service': float}

COMMENT_CONTENT = '太难吃了'

# 使用
# result = ct.llm(text=COMMENT_CONTENT,
# 或
result = ct.llm(text=COMMENT_CONTENT,
                prompt=PROMPT,
                base_url=BASE_URL,
                api_key=API_KEY,
                model_name=MODEL_NAME,
                temperature=0,
                output_format=OUTPUT_FORMAT)

result
```

Run

```
{'taste': -1.0, 'speed': 0.0, 'service': 0.0}
```

<br>

批量运算

```python
import pandas as pd
import cntext as ct


# 构造实验数据
data = ['速度非常快，口味非常好， 服务非常棒！',
        '送餐时间还是比较久',
        '送单很快，菜也不错赞',
        '太难吃了']
df = pd.DataFrame(data, columns=['comment'])


# 分析函数
def llm_analysis(text):
    result = ct.llm(text=text,
                    prompt= '从口味taste、速度speed、服务service三个维度， 对外卖评论内容进行文本分析， 分别返回不同维度的分值(分值范围-1.0 ~ 1.0)',
                    base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
                    api_key='你的API-KEY',
                    model_name='qwen-max',
                    output_format={'taste': float, 'speed': float, 'service': float}
                               )
    return pd.Series(result)


# 批量运算
df2 = df['comment'].apply(llm_analysis)
res_df = pd.concat([df, df2], axis=1)
# 保存分析结果
res_df.to_csv('result.csv', index=False)
res_df
```

![](img/28-llm-analysis.png)

<br>

LLM 更多详细内容，请阅读 [**教程 | 使用在线大模型将文本数据转化为结构化数据**](https://textdata.cn/blog/2025-02-14-using-online-large-model-api-to-transform-text-data-into-structured-data/)

<br>

### 6.2 内置prompt
cntext
```python
ct.llm.tasks_list()
```
Run
```
['sentiment',
 'emotion',
 'classify',
 'intent',
 'keywords',
 'entities',
 'summarize',
 'rewrite',
 'quality',
 'similarity']
```

<br>

```python
# 获取sentiment模板
ct.llm.tasks_get('sentiment')
```
Run
```
{'prompt': '分析评论的情感倾向：返回情感类别 label（pos 表示正面，neg 表示负面，neutral 表示中性）和情感分值 score（取值范围 -1~1，负数为负面）',
 'output_format': {'label': 'str', 'score': 'float'}}
```

<br>


```python
# 使用sentiment提示词模板。
# 启用Ollama服务，调用qwen2.5:7b模型
ct.llm("服务很棒！", task="sentiment", backend="ollama",  model_name="qwen2.5:7b")
```
Run
```
[cntext2x] ✅ 连接模型服务: http://127.0.0.1:11434/v1
{'label': 'pos', 'score': 0.8}
```




<br>

LLM更多详细内容，请阅读  [**教程 | 使用大模型将文本数据转化为结构化数据(阿里云百炼)**](https://textdata.cn/blog/2025-09-12-text-analysis-with-qwen-and-cntext/)

