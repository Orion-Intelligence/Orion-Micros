import requests
import stem as stem
from requests.adapters import HTTPAdapter
from urllib3 import Retry
from stem import Signal
from stem.control import Controller
from crawler.crawler_services.log_manager.log_controller import log
from crawler.crawler_services.redis_manager.redis_enums import REDIS_COMMANDS
from crawler.constants.keys import TOR_KEYS
from crawler.constants.constant import CRAWL_SETTINGS_CONSTANTS
from crawler.crawler_services.shared.env_handler import env_handler
from crawler.crawler_instance.proxies.tor_controller.tor_enums import TOR_CONTROL_PROXIES, TOR_PROXIES, TOR_COMMANDS
from crawler.crawler_services.redis_manager.redis_controller import redis_controller
from crawler.crawler_services.request_manager.request_handler import request_handler


class tor_controller(request_handler):
    __instance = None
    __m_controller = []
    __m_session = None
    __redis_controller = None

    @staticmethod
    def get_instance():
        if tor_controller.__instance is None:
            tor_controller()
        return tor_controller.__instance

    def __init__(self):
        tor_controller.__instance = self
        self.m_request_index = 0
        self.__on_init()

    def __on_init(self):
        self.__redis_controller = redis_controller()

        self.__session = requests.Session()
        retries = Retry(total=1, backoff_factor=0.1, status_forcelist=[500, 502, 503, 504])
        adapter = requests.adapters.HTTPAdapter(pool_connections=1000, pool_maxsize=1000, max_retries=retries)
        self.__session.mount('http://', adapter)

        for connection_controller in TOR_CONTROL_PROXIES:
            m_temp_controller = Controller(
                stem.socket.ControlPort(connection_controller["proxy"], connection_controller["port"]))
            m_temp_controller.authenticate(env_handler.get_instance().env('TOR_PASSWORD'))
            self.__m_controller.append(m_temp_controller)
            self.__invoke_new_circuit(m_temp_controller)

    def __invoke_new_circuit(self, m_temp_controller):
        try:
            m_temp_controller.signal(Signal.NEWNYM)
            self.verify_new_circuit(m_temp_controller)
        except Exception as ex:
            log.g().i("Exception in creeating new circuit")
            log.g().i(ex)

    @staticmethod
    def verify_new_circuit(controller):
        try:
            circuits = controller.get_info("circuit-status")
            log.g().i("Active circuits:")
            for circuit in circuits.splitlines():
                log.g().i(circuit)
        except Exception as ex:
            log.g().i(f"Failed to retrieve circuit status: {ex}")

    def get_non_bootstrapped_tor_instances(self):
        non_bootstrapped_instances = []
        for index, controller in enumerate(self.__m_controller):
            try:
                status = controller.get_info("status/bootstrap-phase")
                if "100%" not in status and "Done" not in status:
                    proxy_ip_port = TOR_PROXIES[index]["http"]
                    current_phase = status.split(" ")[-1]
                    print(f"Tor instance {proxy_ip_port} is in phase: {current_phase}")
                    non_bootstrapped_instances.append((proxy_ip_port, current_phase))
            except Exception as ex:
                print(f"Error checking bootstrap status for controller {index}: {ex}")
                continue
        return non_bootstrapped_instances

    def __on_create_session(self, p_tor_based):
        headers = {
            TOR_KEYS.S_USER_AGENT: CRAWL_SETTINGS_CONSTANTS.S_USER_AGENT,
            'Cache-Control': 'no-cache'
        } if p_tor_based else {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36',
            'Cache-Control': 'no-cache'
        }

        return self.__session, headers

    def __on_proxy(self):
        current_index = int(
            self.__redis_controller.invoke_trigger(REDIS_COMMANDS.S_GET_INT, ["tor_queue_index", -1, None])) + 1
        self.__redis_controller.invoke_trigger(REDIS_COMMANDS.S_SET_INT, ["tor_queue_index", current_index, None])
        return TOR_PROXIES[current_index % len(TOR_PROXIES)], current_index % len(TOR_PROXIES)

    def invoke_trigger(self, p_command, p_data=None):
        if p_command == TOR_COMMANDS.S_CREATE_SESSION:
            return self.__on_create_session(p_data[0])
        elif p_command == TOR_COMMANDS.S_PROXY:
            return self.__on_proxy()
        return None
