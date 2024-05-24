import concurrent.futures
from functools import partial
from typing import List, Tuple

import pandas as pd
from pydantic import BaseModel, Field
from pydantic.fields import FieldInfo

from utils import gen_fields_by_json, get_type_from_str, qa

# load an Excel file to dataframe
df = pd.read_excel('test_data/销售数据.xlsx')


# make fields with spec
class CustomField(BaseModel):
    name: str = Field(..., description='字段名称')
    annotation: str = Field(..., description='字段类型, python 类型')
    comment: str = Field(..., description='字段注释')

    def to_field_info_tuple(self) -> Tuple[str, FieldInfo]:
        return self.name, FieldInfo(
            annotation=get_type_from_str(self.annotation),
            description=self.comment)


class CustomFields(BaseModel):
    fields: List[CustomField] = Field(..., description='新增字段列表')

    def to_field_info_list(self) -> List[Tuple[str, FieldInfo]]:
        return [field.to_field_info_tuple() for field in self.fields]


prompt = """
# 商品的字段：
# name: 商品名称, 无则为`未知`
# num: 数量
# type: 1:赞;2:踩
"""
custom_fields = gen_fields_by_json(
    CustomFields,
    'fields',
    msgs=[prompt],
    sys_msg='根据需求，生成字段信息',
    llm_model='gpt-4o')
custom_fields = CustomFields.model_validate(custom_fields.model_dump())

# run row inference
row_info_str = repr(df.iloc[0])
filling = gen_fields_by_json(
    BaseModel,
    extra_fields=custom_fields.to_field_info_list(),
    msgs=[repr(df.iloc[0])],
    sys_msg='根据需求，生成字段信息',
    llm_model='gpt-4o'
)

# batch run row inference
test_df = df.head(6)
func_lst = [partial(gen_fields_by_json, BaseModel,
                    extra_fields=custom_fields.to_field_info_list(),
                    msgs=[repr(row)],
                    sys_msg='根据需求，生成字段信息',
                    llm_model='gpt-4o')
            for i, row in test_df.iterrows()]
fillings = []
with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = [executor.submit(func) for func in func_lst]
    for future in concurrent.futures.as_completed(futures):
        fillings.append(future.result())

# append result to dataframe
extra_df = pd.DataFrame([f.model_dump() for f in fillings])

new_df = pd.concat([test_df, extra_df], axis=1)

# make code with spec and dataframe info
code_prompt = """
# DataFrame 表信息描述，df变量已经加载无需重复加载
>>> df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 47 entries, 0 to 46
Data columns (total 13 columns):
 #   Column  Non-Null Count  Dtype  
---  ------  --------------  -----  
 0   订单类型    47 non-null     object 
 1   评价来源    47 non-null     object 
 2   品牌      47 non-null     object 
 3   门店编码    47 non-null     object 
 4   门店名称    47 non-null     object 
 5   城市名称    47 non-null     object 
 6   商品详情    47 non-null     object 
 7   下单时间    0 non-null      float64
 8   评价类型    47 non-null     object 
 9   综合评价得分  47 non-null     float64
 10  评分详情    47 non-null     object 
 11  评价内容    36 non-null     object 
 12  评价时间    47 non-null     object 
dtypes: float64(2), object(11)
memory usage: 4.9+ KB

# 生成代码处理Dataframe信息，进行以下统计：
name: 商品名称，对应`商品详情`列，类型为字段字符串，例如'["其他", "霸气杨枝甘露"]', 可通过json.loads转换成列表
num: 数量，累计数量
type: 评级，1:赞;2:踩，对应`评价类型`列，值为`好评`或`差评`
"""
answer_str = qa(code_prompt)
# get code from answer
code = answer_str.split('```python')[1].split('```')[0].strip()
func_str = qa(f'Turn the following code into a function:\n ```python\n{code}\n```')
func_str_final = func_str.split('```python')[1].split('```')[0].strip()
# todo: extract the function name

# run the code
exec(func_str_final, globals())
result = analyze_product_reviews(df)
print(result.to_markdown())
