from pydantic import BaseModel, Field, field_validator, model_validator
from datetime import date, datetime
from typing import List, Optional
from crawler.constants.enums import VALID_NETWORK_TYPES, VALID_CONTENT_TYPES


class leak_model(BaseModel):
    m_title: str
    m_url: str
    m_base_url: str
    m_content: str
    m_important_content: str
    m_network: str
    m_section: Optional[List[str]] = Field(default_factory=list)
    m_content_type: List[str] = Field(default_factory=list)
    m_screenshot: str
    m_weblink: List[str] = Field(default_factory=list)
    m_dumplink: List[str] = Field(default_factory=list)
    m_websites: List[str] = Field(default_factory=list)
    m_logo_or_images: List[str] = Field(default_factory=list)
    m_leak_date: Optional[date] = None
    m_data_size: Optional[str] = None
    m_revenue: Optional[str] = None

    @field_validator('m_leak_date', mode='before')
    def parse_leak_date(cls, value):
        if isinstance(value, str):
            try:
                return datetime.strptime(value, "%Y-%m-%d").date()
            except ValueError:
                raise ValueError(f"Invalid date format for m_leak_date: {value}. Expected format: YYYY-MM-DD.")
        return value

    @model_validator(mode='after')
    def check_required_fields_and_enums(self):
        required_fields = ["m_title", "m_url", "m_content", "m_base_url", "m_important_content", "m_screenshot"]
        for field_name in required_fields:
            if getattr(self, field_name) is None:
                raise ValueError(f"The field '{field_name}' is required and cannot be None.")

        if self.m_network not in VALID_NETWORK_TYPES:
            raise ValueError(
                f"Invalid network type provided: {self.m_network}. Must be one of {', '.join(VALID_NETWORK_TYPES)}.")

        if not isinstance(self.m_content_type, list):
            raise ValueError("m_content_type must be a list of valid content types.")

        if not all(content in VALID_CONTENT_TYPES for content in self.m_content_type):
            raise ValueError(
                f"Invalid content type(s) provided: {self.m_content_type}. Must be a subset of {', '.join(VALID_CONTENT_TYPES)}.")

        return self

    model_config = {
        "arbitrary_types_allowed": True,
        "json_encoders": {
            date: lambda v: v.strftime("%Y-%m-%d") if v else None
        }
    }
