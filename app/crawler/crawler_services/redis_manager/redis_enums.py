from enum import Enum
from crawler.crawler_services.shared.env_handler import env_handler


class REDIS_CONNECTIONS:
    S_DATABASE_IP = 'redis_server'
    S_DATABASE_PORT = 6379
    S_DATABASE_PASSWORD = env_handler.get_instance().env('REDIS_PASSWORD')


class REDIS_KEYS:
    RAW_HTML_SCORE = "RAW_HTML_SCORE_"
    RAW_HTML_CODE = "RAW_HTML_CODE_"
    LEAK_PARSED = "LEAK_PARSED_CODE_"
    HOST_FAILURE_COUNT = "HOST_FAIL_"
    HOST_LOW_YIELD_COUNT = "LOW_YIELD_"
    UNIQIE_CRAWLER_RUNNING = "UNIQIE_CRAWLER_RUNNING"


class CUSTOM_SCRIPT_REDIS_KEYS(Enum):
    URL_PARSED = "URL_PARSED_"


class REDIS_COMMANDS:
    S_SET_BOOL = 1
    S_GET_BOOL = 2
    S_SET_INT = 3
    S_GET_INT = 4
    S_SET_STRING = 5
    S_GET_STRING = 6
    S_SET_LIST = 7
    S_GET_LIST = 8
    S_GET_KEYS = 9
    S_GET_FLOAT = 10
    S_SET_FLOAT = 11
    S_FLUSH_ALL = 12
    S_ACQUIRE_LOCK = 13
    S_RELEASE_LOCK = 14
