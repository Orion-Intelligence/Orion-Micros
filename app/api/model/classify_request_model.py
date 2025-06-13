from pydantic import BaseModel


class classify_request_model(BaseModel):
    title: str
    description: str
    keyword: str
