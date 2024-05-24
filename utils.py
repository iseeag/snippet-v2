from io import BytesIO
from typing import (List, Literal, Tuple, Type, TypeVar, Union, get_args,
                    get_origin)

import pandas as pd
from openai import ChatCompletion, OpenAI
from pydantic import create_model
from pydantic.fields import FieldInfo
from pydantic.main import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

T = TypeVar("T", bound=BaseModel)

oai: OpenAI = OpenAI(base_url='https://key.agthub.ai/v1',
                     api_key='sk-6jpejlScJXs1zpv6369841E107C54d999dEc953b55350343')


def get_type_from_str(type_string: str, context: dict = None) -> Type:
    specified_type = eval(type_string, context)
    return specified_type


def oai_chat_completion_create(**kwargs) -> ChatCompletion:
    resp = oai.chat.completions.create(**kwargs)
    return resp


class Chat:
    def __init__(self, sys_msg, token_limit=10000, use_context=True, n_context_msg=999):
        self.sys_msg = sys_msg
        self.history = []
        self.token_limit = token_limit
        self.use_context = use_context
        self.n_context_msg = n_context_msg

    def __call__(
            self,
            *user_msgs,
            custom_msgs: List = None,
            sys_msg: str = None,
            model: str = 'gpt-4o',
            **kwargs
    ):
        sys_msg_str = sys_msg or self.sys_msg
        llm_model = model
        # llm_model = 'gpt-3.5-turbo-0125', 'gpt-4o-turbo-preview',

        sys_msg = {'role': 'system', 'content': sys_msg_str}
        user_msgs = [{'role': 'user', 'content': msg} for msg in user_msgs]
        context_msgs = self.history[:self.n_context_msg] if self.use_context else []
        custom_msgs = [m.model_dump() for m in custom_msgs] if custom_msgs else []
        messages = [
            *user_msgs,
            *custom_msgs
        ]
        print(f"system: {sys_msg_str}")
        for message in messages:
            print(f"{message['role']}: {message['content']}")
        resp = oai_chat_completion_create(
            model=llm_model,
            temperature=0.0,
            messages=[sys_msg] + context_msgs + messages,
            seed=42,
            **kwargs
        )
        print(f'LLM <{llm_model}> fingerprint: {resp.system_fingerprint}')

        assistant_msg = resp.choices[0].message.content
        print(assistant_msg)
        print(f'completion tokens: {resp.usage}')
        self.history.extend(messages)
        self.history.append({'role': 'assistant', 'content': assistant_msg})
        return assistant_msg


qa = Chat(sys_msg="你什么都能做", use_context=False)


def schema_to_json_description(model: Type[T], indent=4) -> str:
    schema = model.schema()
    DEFAULT = '默认'
    REQUIRED = '必填'

    def unpack(obj, definitions):
        if not definitions:
            return obj
        if obj.get('$ref'):
            ref = obj['$ref']
            return definitions[ref.split('/')[-1]]
        return obj

    def process_enum(obj):
        if obj.get('enum') and obj.get('description'):
            description = obj.get('description')
            return {k: v for k, v in obj.items() if k != 'description'}, description
        return obj, ''

    type_print_map = {type('str'): 'string',
                      type(1): 'integer'}

    def serialize(obj, level, definitions, is_required=False, comma=False):
        indent_str = ' ' * (level * indent)
        item_indent_str = ' ' * ((level + 1) * indent)
        comment = ''
        obj_description = obj.get('description', '')
        default_str = f'{DEFAULT}: {obj.get("default", "")}' if obj.get('default') else ''
        is_required_str = REQUIRED if is_required else ''
        comment_str_lst = [obj_description, default_str, is_required_str]
        comma_str = ',' if comma else ''
        if obj_description and obj.get('anyOf') and {'type': 'null'} in obj['anyOf']:
            obj['anyOf'].remove({'type': 'null'})
            comment_str_lst.append('can be null')
        if any(comment_str_lst):
            comment = f'  // ' + ', '.join([s for s in comment_str_lst if s]) + '.'

        if obj.get('properties'):
            properties = obj['properties']
            required = obj.get('required', [])
            items = []
            n_items = len(properties)
            for i, (k, v) in enumerate(properties.items()):
                comma = True if i < n_items - 1 else False
                value_str = serialize(unpack(v, definitions),
                                      level + 1,
                                      definitions,
                                      k in required,
                                      comma=comma)
                item = (f'\n{item_indent_str}"{k}": '
                        f'{value_str}')
                items.append(item)
            return f"{{{comment}" + "".join(items) + f'\n{indent_str}}}'
        if obj.get('enum'):
            obj_type = obj.get("type", type_print_map.get(type(obj['enum'][0]), 'obj'))
            return f'{obj_type}{comma_str}{comment}'
        if obj.get('type') == 'array':
            sub_obj = unpack(obj.get("items"), definitions)
            items = f'\n{item_indent_str}{serialize(sub_obj, level + 1, definitions)}'
            return "[" + items + f'\n{indent_str}]{comma_str}{comment}'
        if obj.get('type') == 'string':
            return f'string{comma_str}{comment}'
        if obj.get('type') == 'boolean':
            return f'boolean{comma_str}{comment}'
        if obj.get('type') == 'integer':
            return f'integer{comma_str}{comment}'
        if obj.get('anyOf'):
            return f' or '.join([
                serialize(unpack(o, definitions), level, definitions, is_required)
                for o in obj['anyOf']]) + comma_str + f'{comment}'
        if obj.get('allOf'):
            assert len(obj['allOf']) == 1
            sub_obj = unpack(obj['allOf'][0], definitions)
            sub_obj, extra_comment = process_enum(sub_obj)
            item = serialize(sub_obj, level, definitions)
            return item + comma_str + f'{comment} {extra_comment}'

        return f'obj{comma_str}{comment}'

    return serialize(schema, 0, schema.get('$defs', {}))


def remove_optional(field: FieldInfo) -> Tuple[Type, FieldInfo]:
    if get_origin(field.annotation) is Union:
        inner_types = get_args(field.annotation)
        inner_types = [t for t in inner_types if t is not type(None)]
        annotation = Union[*inner_types]
        new_field = FieldInfo(annotation=annotation, description=field.description)
        return annotation, new_field
    return field.annotation, field


def synthesize_model_by_fields(
        model: Type[T],
        *fields: str,
        extra_fields: List[Tuple[str, FieldInfo]] = None
) -> Type[T]:
    model_fields = extra_fields or []
    model_fields.extend([(field, model.model_fields[field]) for field in fields])
    return create_model(
        model.__name__,
        **{field: remove_optional(f_info)
           for field, f_info in model_fields}
    )


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def gen_fields_by_json(
        model: Type[T],
        *fields: str,
        extra_fields: List[Tuple[str, FieldInfo]] = None,
        msgs: List[str],
        sys_msg: str,
        llm_model: Literal["gpt-3.5-turbo", "gpt-4-turbo-preview", "gpt-4o"] = "gpt-3.5-turbo"
) -> T:
    model = synthesize_model_by_fields(model, *fields, extra_fields=extra_fields)
    json_gen_spec = schema_to_json_description(model)
    json_gen_spec_msg = f"json spec:\n{json_gen_spec}"
    obj_json = qa(
        *msgs,
        json_gen_spec_msg,
        sys_msg=sys_msg,
        model=llm_model,
        response_format={"type": "json_object"}
    )
    return model.model_validate_json(obj_json)


def to_excel(df: pd.DataFrame):
    output = BytesIO()
    writer = pd.ExcelWriter(output)
    df.to_excel(writer, index=False, sheet_name='Sheet1')
    writer.close()
    processed_data = output.getvalue()
    return processed_data
