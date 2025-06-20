import importlib
import json
import os
import sys

from playwright.async_api import async_playwright
from typing import Optional

from api.model.rule_model import FetchProxy
from api.runtime_parse_manager.runtime_parse_enum import RUNTIME_PARSE_REQUEST_QUERIES, RUNTIME_PARSE_REQUEST_COMMANDS
from crawler.crawler_instance.proxies.tor_controller.tor_controller import tor_controller
from crawler.crawler_instance.proxies.tor_controller.tor_enums import TOR_COMMANDS
from crawler.crawler_services.log_manager.log_controller import log


class runtime_parse_controller:

    def __init__(self):
        self.module_cache = {}
        self.driver = None
        self.playwright = None
        self.browser = None

    @staticmethod
    async def _get_block_resources(route):
        request_url = route.request.url.lower()

        if any(request_url.startswith(scheme) for scheme in ["data:image", "data:video", "data:audio"]) or \
                route.request.resource_type in ["image", "media", "font", "stylesheet"]:
            return await route.abort()
        else:
            return await route.continue_()

    async def _initialize_webdriver(self, use_proxy: FetchProxy = FetchProxy.TOR) -> Optional[object]:

        tor_proxy = None
        if use_proxy == FetchProxy.TOR:
            tor_proxy, _ = tor_controller.get_instance().invoke_trigger(TOR_COMMANDS.S_PROXY, [])

        proxy_url = next(iter(tor_proxy.values()))
        ip_port = proxy_url.split('//')[1]
        ip, port = ip_port.split(':')
        proxy_host = tor_proxy.get('host', ip)
        proxy_port = tor_proxy.get('port', port)
        proxies = {"server": f"socks5://{proxy_host}:{proxy_port}"}

        if self.playwright is None:
            self.playwright = await async_playwright().start()

        if self.browser is None:
            self.browser = await self.playwright.chromium.launch(headless=True, proxy=proxies)

        context = await self.browser.new_context()
        context.set_default_timeout(600000)
        context.set_default_navigation_timeout(600000)
        await context.route("**/*", self._get_block_resources)
        return context

    # @staticmethod
    # async def _initialize_webdriver(use_proxy: bool = True) -> Optional[object]:
    #     if use_proxy:
    #         tor_proxy = "socks5://127.0.0.1:9150"
    #         playwright = await async_playwright().start()
    #         browser = await playwright.chromium.launch(headless=False,
    #                                                    proxy={"server": tor_proxy} if tor_proxy else None)
    #     else:
    #         playwright = await async_playwright().start()
    #         browser = await playwright.chromium.launch(headless=False)
    #
    #     context = await browser.new_context()
    #     return context

    async def get_email_username(self, query):
        result = []
        try:
            if self.driver is None:
                self.driver = await self._initialize_webdriver()
        except Exception as ex:
            log.g().i(ex)
            return json.dumps(result)

        for parser in RUNTIME_PARSE_REQUEST_QUERIES.S_USERNAME:
            try:
                parse_script = self.on_init_leak_parser(parser)
                query["url"] = parse_script.base_url
                response = await parse_script.parse_leak_data(query, self.driver)
                if len(response.cards_data) > 0:
                    result.append(response.model_dump())
            except Exception as ex:
                log.g().i(ex)
                self.driver = None
                if self.browser:
                    await self.browser.close()
                    self.browser = None

        return json.dumps(result)

    def on_init_leak_parser(self, file_name):
        class_name = "_" + file_name
        try:
            module_path = f"raw.parsers.api_collector.{class_name}"
            parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            if parent_dir not in sys.path:
                sys.path.append(parent_dir)
            if class_name not in self.module_cache:
                module = importlib.import_module(module_path)
                class_ = getattr(module, class_name)
                self.module_cache[class_name] = class_()
            return self.module_cache[class_name]

        except Exception as ex:
            log.g().i(ex)
            return None

    async def invoke_trigger(self, command, data=None):
        if command == RUNTIME_PARSE_REQUEST_COMMANDS.S_PARSE_USERNAME:
            return await self.get_email_username(data)
        return None

    # async def main():
    #     url = "http://breachdbsztfykg2fdaq2gnqnxfsbj5d35byz3yzj73hazydk4vq72qd.onion/"
    #     email = "msmannan00@gmail.com"
    #     username = "msmannan00"
    #     query = {"url": url, "email": email, "username": username}
    #
    #     try:
    #         result = await runtime_parse_controller().invoke_trigger(RUNTIME_PARSE_REQUEST_COMMANDS.S_PARSE_USERNAME, query)
    #         print(result)
    #     except Exception as e:
    #         print("Error occurred:", e)
    #     finally:
    #         pass
    #
    #
    # if __name__ == "__main__":
    #     asyncio.run(main())
