from pydantic import BaseModel


class parse_request_model(BaseModel):
    text: str