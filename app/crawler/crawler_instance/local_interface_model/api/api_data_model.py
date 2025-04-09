from pydantic import BaseModel, Field, SkipValidation
from typing import List
from crawler.crawler_instance.local_shared_model.data_model.leak_model import leak_model


class api_data_model(BaseModel):
    cards_data: List[SkipValidation[leak_model]] = Field(default_factory=list)
    base_url: str = ""
    m_network: str = ""
    content_type: List[str] = []
