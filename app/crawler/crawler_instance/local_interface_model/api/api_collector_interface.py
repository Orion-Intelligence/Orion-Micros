from abc import ABC, abstractmethod
from typing import Dict, List

from playwright.async_api import BrowserContext

from crawler.crawler_instance.local_shared_model.rule_model import RuleModel


class api_collector_interface(ABC):
    @property
    @abstractmethod
    def base_url(self) -> str:
        """Return the base URL."""
        pass

    @property
    @abstractmethod
    def rule_config(self) -> RuleModel:
        """Return the rule configuration."""
        pass

    @abstractmethod
    def parse_leak_data(self, query: Dict[str, str], context: BrowserContext):
        pass
