import concurrent.futures
from functools import partial
from typing import List, Tuple

import pandas as pd
from pydantic import BaseModel, Field
from pydantic.fields import FieldInfo

from utils import gen_fields_by_json, get_type_from_str, qa


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


def create_custom_fields(prompt: str) -> CustomFields:
    custom_fields = gen_fields_by_json(
        CustomFields,
        'fields',
        msgs=[prompt],
        sys_msg='根据需求，生成字段信息',
        llm_model='gpt-4o')
    custom_fields = CustomFields.model_validate(custom_fields.model_dump())
    return custom_fields


def batch_row_inference(df: pd.DataFrame, custom_fields: CustomFields, max_row: int = 100) -> pd.DataFrame:
    df = df.head(max_row)
    func_lst = [partial(gen_fields_by_json, BaseModel,
                        extra_fields=custom_fields.to_field_info_list(),
                        msgs=[repr(row)],
                        sys_msg='根据需求，生成字段信息',
                        llm_model='gpt-4o')
                for i, row in df.iterrows()]
    fillings = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(func) for func in func_lst]
        for future in concurrent.futures.as_completed(futures):
            fillings.append(future.result())

    extra_df = pd.DataFrame([f.model_dump() for f in fillings])
    return pd.concat([df, extra_df], axis=1)


def df_inference(df: pd.DataFrame, prompt: str, max_row: int = 50) -> pd.DataFrame:
    custom_fields = create_custom_fields(prompt)
    return batch_row_inference(df, custom_fields, max_row)
