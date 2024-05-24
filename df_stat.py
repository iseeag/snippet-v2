import re
from io import StringIO
from typing import Dict, Tuple

import pandas as pd
from openai import OpenAI


def create_sheet_description(index: int, file_name: str, var_name: str, df: pd.DataFrame):
    buffer = StringIO()
    df.info(buf=buffer, verbose=True)
    info_str = buffer.getvalue()
    buffer.close()
    return f"""{index}. {var_name}: DataFrame  <{file_name}>
{info_str}
"""


def create_sheet_var_comment(file_name: str, var_name: str):
    return f"""# ‘{file_name}’表格对象
{var_name}: pd.DataFrame
"""


def create_prompt(query: str, sheet_dict: Dict[str, Tuple[str, pd.DataFrame]]):
    df_descriptions = [create_sheet_description(i, file_name, var_name, df)
                       for i, (var_name, (file_name, df)) in enumerate(sheet_dict.items(), 1)]
    df_dsrp_str = '\n'.join(df_descriptions)
    df_var_comments = [create_sheet_var_comment(file_name, var_name)
                       for var_name, (file_name, _) in sheet_dict.items()]
    df_var_comments_str = '\n'.join(df_var_comments)
    prompt = f"""
# 任务描述
1. 你根据提问的内容，从以下提供的 DataFrame 表信息，编写一段 python 代码来实现
2. 以下提供的表格已经读取成 python 的 DataFrame 对象，可以直接操作

# DataFrame 表信息描述
{df_dsrp_str}

# 提问内容
{query}

# 代码生成
```python
import pandas as pd

{df_var_comments_str}

# 补充实现代码
# 打印输出的结果（结果数据框命名为 df_result）
```
"""
    return prompt


def create_code(prompt: str):
    # 模型请求
    client = OpenAI(
        base_url='https://key.agthub.ai/v1',
        api_key='sk-6jpejlScJXs1zpv6369841E107C54d999dEc953b55350343'
    )

    completion = client.chat.completions.create(
        model='gpt-4',
        temperature=1e-3,
        messages=[{'role': 'user', 'content': prompt}],
    )
    content = completion.choices[0].message.content
    match_code = re.search(r'```python(.*?)```', content, re.DOTALL)
    python_code = match_code.group(1)
    print(python_code)
    return python_code
