from typing import List
from pydantic import BaseModel, Field
from crawler.crawler_instance.local_shared_model.data_model.leak_model import leak_model
from crawler.crawler_instance.local_shared_model.data_model.defacement_model import defacement_model

class leak_data_model(BaseModel):
    cards_data: List[leak_model | defacement_model] = Field(default_factory=list)
    contact_link: str = ""
    base_url: str = ""
    seed_url: str = ""
    m_network: str = ""
    content_type: List[str] = Field(default_factory=list)
