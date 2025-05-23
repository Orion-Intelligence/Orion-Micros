from crawler.constants.constant import RAW_PATH_CONSTANTS, TOR_CONSTANTS


class TOR_COMMANDS:
  S_START = 1
  S_RESTART = 2
  S_GENERATED_CIRCUIT = 3
  S_RELEASE_SESSION = 4
  S_CREATE_SESSION = 5
  S_PROXY = 6


class TOR_CMD_COMMANDS:
  S_START_DIRECT = RAW_PATH_CONSTANTS.S_SIGWIN_PATH + " " + TOR_CONSTANTS.S_SHELL_CONFIG_PATH + " " + TOR_CONSTANTS.S_TOR_PATH + " " + "build-start-tor"
  S_START_DOCKERISED = "." + TOR_CONSTANTS.S_SHELL_CONFIG_PATH + " " + TOR_CONSTANTS.S_TOR_PATH + " " + "build-start-tor"


class TOR_STATUS:
  S_RUNNING = 1
  S_PAUSE = 2
  S_STOP = 3
  S_READY = 4
  S_START = 5
  S_CLOSE = 6


TOR_PROXIES = [
  {
    "http": "socks5h://172.15.0.10:" + "9352",
    "https": "socks5h://172.15.0.10:" + "9052"
  }
  # ,
  # {
  #   "http": "socks5h://172.15.0.11:" + "9354",
  #   "https": "socks5h://172.15.0.11:" + "9354"
  # },
  # {
  #   "http": "socks5h://172.15.0.12:" + "9356",
  #   "https": "socks5h://172.15.0.12:" + "9356"
  # },
  # {
  #   "http": "socks5h://172.15.0.13:" + "9358",
  #   "https": "socks5h://172.15.0.13:" + "9358"
  # },
  # {
  #   "http": "socks5h://172.15.0.14:" + "9360",
  #   "https": "socks5h://172.15.0.14:" + "9360"
  # },
  # {
  #   "http": "socks5h://172.15.0.15:" + "9362",
  #   "https": "socks5h://172.15.0.15:" + "9362"
  # },
  # {
  #   "http": "socks5h://172.15.0.16:" + "9364",
  #   "https": "socks5h://172.15.0.16:" + "9364"
  # },
  # {
  #   "http": "socks5h://172.15.0.17:" + "9366",
  #   "https": "socks5h://172.15.0.17:" + "9366"
  # },
  # {
  #   "http": "socks5h://172.15.0.18:" + "9368",
  #   "https": "socks5h://172.15.0.18:" + "9368"
  # },
  # {
  #   "http": "socks5h://172.15.0.19:" + "9370",
  #   "https": "socks5h://172.15.0.19:" + "9370"
  # }
]
TOR_CONTROL_PROXIES = [
  {
    "proxy": "172.15.0.10",
    "port": 9353
  }
  # ,
  # {
  #   "proxy": "172.15.0.11",
  #   "port": 9355
  # },
  # {
  #   "proxy": "172.15.0.12",
  #   "port": 9357
  # },
  # {
  #   "proxy": "172.15.0.13",
  #   "port": 9359
  # },
  # {
  #   "proxy": "172.15.0.14",
  #   "port": 9361
  # },
  # {
  #   "proxy": "172.15.0.15",
  #   "port": 9363
  # },
  # {
  #   "proxy": "172.15.0.16",
  #   "port": 9365
  # },
  # {
  #   "proxy": "172.15.0.17",
  #   "port": 9367
  # },
  # {
  #   "proxy": "172.15.0.18",
  #   "port": 9369
  # },
  # {
  #   "proxy": "172.15.0.19",
  #   "port": 9371
  # }
]
