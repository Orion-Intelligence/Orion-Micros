from enum import Enum

class FetchConfig(str, Enum):
    SELENIUM = "selenium"
    REQUESTS = "requests"

class ThreatType(str, Enum):
    DEFACEMENT = "defacement_collector"
    API = "api_collector"
    LEAK = "leak_collector"

class FetchProxy(str, Enum):
    TOR = "tor"
    NONE = "none"

class RuleModel:
    def __init__(self, m_timeout: int = 17200, m_fetch_config: FetchConfig = FetchConfig.SELENIUM, m_fetch_proxy: FetchProxy = FetchProxy.NONE, m_resoource_block = True, m_threat_type=ThreatType.LEAK):
        self.m_timeout = m_timeout
        self.m_fetch_config = m_fetch_config
        self.m_fetch_proxy = m_fetch_proxy
        self.m_resoource_block = m_resoource_block
        self.m_threat_type = m_threat_type

