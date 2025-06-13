from pydantic import BaseModel


class parse_cti_model(BaseModel):
    text: str


class parse_request_model(BaseModel):
    data: str
