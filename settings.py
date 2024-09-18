from pydantic import BaseModel


class Config(BaseModel):
    MAX_ROW_INFER: int = 500
    oai_model: str = 'gpt-4o'
    oai_key: str = 'sk-6jpejlScJXs1zpv6369841E107C54d999dEc953b55350343'
    oai_url: str = 'https://key.agthub.ai/v1'


config = Config()
