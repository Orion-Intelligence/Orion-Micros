from pydantic import BaseModel, Field, field_validator
from datetime import date, datetime
from typing import Optional, List

class defacement_model(BaseModel):
    m_location: List[str]
    m_attacker: List[str]
    m_team: str
    m_web_server: List[str]
    m_base_url: str
    m_url: str
    m_network: str
    m_content: str
    m_date_of_leak: Optional[date] = None
    m_web_url: List[str]
    m_screenshot: Optional[str] = None
    m_mirror_links: List[str] = Field(default_factory=list)

    @field_validator('m_date_of_leak', mode='before')
    def parse_date_of_leak(cls, value):
        if isinstance(value, str):
            try:
                return datetime.strptime(value, "%Y-%m-%d").date()
            except ValueError:
                raise ValueError(f"Invalid date format for m_date_of_leak: {value}. Expected format: YYYY-MM-DD.")
        return value

    model_config = {
        "arbitrary_types_allowed": True
    }
