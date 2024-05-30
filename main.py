from df_stat import create_code, create_prompt

import pandas as pd
import streamlit as st

from row_infer import df_inference
from utils import to_excel, qa

MAX_ROW_INFER = 30

uploaded_files = st.file_uploader("Excel上传", type=['xlsx'], accept_multiple_files=True)
sheet_dict = {}  # {'df_n': (sheet_name, df)}
placeholder = st.empty()
if uploaded_files is not None:
    for i, uploaded_file in enumerate(uploaded_files):
        df = pd.read_excel(uploaded_file)
        file_name = uploaded_file.name.split('.')[0]
        sheet_dict[f'df_{i}'] = (file_name, df)
        exec(f'df_{i} = df')
        placeholder.write(f'已读表格：[{"、".join([n for n, _ in sheet_dict.values()])}]')
        # st.write(df)

df_result = pd.DataFrame()
default_query = f"""****（二选一）****

------ <汇总统计 [生成代码->执行代码->返回结果] > ------

# 生成代码处理Dataframe信息，进行以下统计：
name: 商品名称，对应`商品详情`列，类型为字段字符串，例如'["其他", "霸气杨枝甘露"]', 可通过json.loads转换成列表
num: 数量，累计数量
type: 评级，1:赞;2:踩，对应`评价类型`列，值为`好评`或`差评`

------ <行推理 [定义字段名称->AI逐行进行推理->返回结果] > ------

# 商品的字段：
name: 商品名称, 无则为`未知`
num: 数量
type: 1:赞;2:踩
*注：行推理token消耗较多，只对前{MAX_ROW_INFER}行做推理
"""
query = st.text_area("需求填写", value=default_query, height=300)
log = ''
if st.button("执行“全局处理”"):
    log = ''
    prompt = create_prompt(query, sheet_dict)
    print(prompt)
    log += f'提示词\n```text{prompt}```\n\n'
    code = create_code(prompt)
    print(code)
    log += f'生成代码\n```python\n{code}```\n\n'
    exec(code)
    st.write(df_result)
    st.write(log)

if st.button(f"执行“行推理”"):
    log = ''
    log += f'提示词\n```text{query}```\n\n'
    df_result = df_inference(sheet_dict['df_0'][1], query, MAX_ROW_INFER)
    st.write(df_result)
    st.write(log)

if st.button(f"执行无状态“问答”"):
    log = ''
    log += f'提示词\n```text{query}```\n\n'
    answer = qa(query)
    st.write(answer)
    st.write(log)


def generate_excel():
    excel_data = to_excel(df_result)
    st.session_state.excel_data = excel_data


if 'excel_data' not in st.session_state:
    st.session_state.excel_data = b''

st.download_button(
    label="下载Excel(可能要点多一次才有数据)",
    data=st.session_state.excel_data,
    file_name='提取关键词.xlsx',
    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    on_click=generate_excel
)
