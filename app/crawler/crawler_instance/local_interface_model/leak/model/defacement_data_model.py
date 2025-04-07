from pydantic import BaseModel, Field, SkipValidation
from typing import List

from crawler.crawler_instance.local_shared_model.data_model.defacement_model import defacement_model


class defacement_data_model(BaseModel):
    cards_data: List[SkipValidation[defacement_model]] = Field(default_factory=list)
    base_url: str = ""
    m_network: str = ""
    content_type: List[str] = []
