import json
import os
import re
import zipfile
import socket
from datetime import datetime
from typing import Optional
from urllib.parse import urlunparse
from bs4 import BeautifulSoup
from crawler.constants.strings import MANAGE_MESSAGES
from crawler.crawler_services.log_manager.log_controller import log
from urllib.parse import urlparse
import base64
import hashlib

STOPWORDS = {
    'a', 'an', 'the', 'and', 'or', 'but', 'if', 'while', 'with', 'at', 'by', 'for',
    'from', 'into', 'on', 'of', 'to', 'in', 'out', 'over', 'under', 'above', 'below',
    'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do',
    'does', 'did', 'can', 'could', 'should', 'would', 'may', 'might', 'must', 'will',
    'shall', 'not', 'no', 'nor', 'this', 'that', 'these', 'those', 'it', 'its', 'i',
    'you', 'he', 'she', 'we', 'they', 'them', 'me', 'my', 'your', 'his', 'her', 'our',
    'their', 'as', 'so', 'than', 'too', 'very'
}


class helper_method:

    @staticmethod
    def generate_data_hash(data):
        if isinstance(data, dict):
            data_copy = {key: value for key, value in data.items() if
                         key not in {'m_update_date', 'm_base_url', 'm_url'}}
            data_string = json.dumps(data_copy, sort_keys=True)
        elif isinstance(data, str):
            data_string = data
        else:
            raise ValueError("Input must be a dictionary or a string")

        return hashlib.sha256(data_string.encode('utf-8')).hexdigest()

    @staticmethod
    def get_base_url(url):
        parsed_url = urlparse(url)
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
        return base_url

    @staticmethod
    def is_stop_word(p_word):
        if p_word in STOPWORDS:
            return True
        else:
            return False

    @staticmethod
    def strip_special_character(p_text):
        m_text = re.sub(r"^\W+", "", p_text)
        return m_text

    @staticmethod
    def on_clean_url(p_url):
        parsed_url = urlparse(p_url)
        netloc = parsed_url.netloc.replace("www.", "", 1)
        cleaned_url = urlunparse((
            parsed_url.scheme,
            netloc.lower(),
            parsed_url.path.rstrip('/ '),
            parsed_url.params,
            parsed_url.query,
            parsed_url.fragment
        ))
        return cleaned_url

    @staticmethod
    def extract_and_convert_date(text: str) -> Optional[datetime.date]:
        for pattern, fmt in [
            (r'(\d{4}-\d{2}-\d{2})', "%Y-%m-%d"),
            (r'(\d{4}/\d{2}/\d{2})', "%Y/%m/%d"),
            (r'(\d{2}-\d{2}-\d{4})', "%d-%m-%Y"),
            (r'(\d{2}/\d{2}/\d{4})', "%m/%d/%Y"),
            (r'(\d{1,2} \w+ \d{4})', "%d %B %Y")
        ]:
            if match := re.search(pattern, text):
                try:
                    return datetime.strptime(match.group(0), fmt).date()
                except ValueError:
                    continue
        return None

    @staticmethod
    def get_screenshot_base64(page, search_string):
        page.wait_for_load_state("load")
        element = page.locator(f":text('{search_string}')").first
        element.evaluate("element => element.scrollIntoView({ block: 'start' })")
        screenshot_bytes = page.screenshot()
        return base64.b64encode(screenshot_bytes).decode('utf-8')

    @staticmethod
    def get_network_type(url: str):
        try:
            if not url.startswith("http"):
                url = "http://" + url
            parsed_url = urlparse(url)
            if not parsed_url.scheme or not parsed_url.netloc:
                return "invalid"
            if re.search(r"\.onion$", parsed_url.netloc, re.IGNORECASE):
                return "onion"
            if re.search(r"\.i2p$", parsed_url.netloc, re.IGNORECASE):
                return "i2p"
            return "clearnet"
        except Exception:
            return "invalid"

    @staticmethod
    def extract_emails(text: str) -> list:

        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        emails = re.findall(email_pattern, text)
        return emails

    @staticmethod
    def extract_phone_numbers(text: str) -> list:
        phone_pattern = r'\(?\b\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}\b'
        phone_numbers = re.findall(phone_pattern, text)

        filtered_phone_numbers = []
        for number in phone_numbers:
            digits_only = re.sub(r'[^0-9]', '', number)

            if 7 <= len(digits_only) <= 15:
                if '(' in text[text.find(number):text.find(number) + len(number)]:
                    filtered_phone_numbers.append(number)
                else:
                    filtered_phone_numbers.append(number)

        return filtered_phone_numbers

    @staticmethod
    def extract_text_from_html(html: str) -> str:
        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text(separator=' ')
        return helper_method.clean_text(text)

    @staticmethod
    def clear_hosts_file(file_path):
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            with open(file_path, 'w'):
                pass

        except Exception:
            pass

    @staticmethod
    def get_host_name(p_url):
        m_parsed_uri = urlparse(p_url)
        m_netloc = m_parsed_uri.netloc

        if m_netloc.startswith('www.'):
            m_netloc = m_netloc[4:]

        netloc_parts = m_netloc.split('.')

        if len(netloc_parts) > 2:
            m_host_name = netloc_parts[-2]
        elif len(netloc_parts) == 2:
            m_host_name = netloc_parts[0]
        else:
            m_host_name = m_netloc

        return m_host_name

    @staticmethod
    def get_class_name(p_url):
        m_parsed_uri = urlparse(p_url)
        m_netloc = m_parsed_uri.netloc
        if m_netloc.startswith('www.'):
            m_netloc = m_netloc[4:]

        netloc_parts = m_netloc.split('.')
        if len(netloc_parts) > 1:
            m_host_name = '.'.join(netloc_parts[:-1])
        else:
            m_host_name = m_netloc
        return m_host_name

    @staticmethod
    def get_service_ip():
        try:
            service_name = os.getenv('SEARCH_SERVICE', 'orion-search-web')
            service_ip = socket.gethostbyname(service_name)
            return f"http://{service_ip}:8080"
        except socket.error as e:
            return f"Error resolving service IP: {e}"

    @staticmethod
    def check_service_status(service_name, host, port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.connect((host, port))
                return True
            except socket.error:
                log.g().e(
                    MANAGE_MESSAGES.S_SERVICE_NOT_INITIATED + " : " + f"{service_name} is not running or not installed.")
                return False

    @staticmethod
    def extract_zip(from_path, to_path):
        os.makedirs(to_path, exist_ok=True)
        try:
            with zipfile.ZipFile(from_path, 'r') as zip_ref:
                zip_ref.extractall(to_path)
        except Exception as e:
            log.g().e(f"Error occurred while extracting {from_path}: {e}")

    @staticmethod
    def split_host_url(p_url):
        m_parsed_uri = urlparse(p_url)
        m_host_url = '{uri.scheme}://{uri.netloc}/'.format(uri=m_parsed_uri)
        if m_host_url.endswith("/"):
            m_host_url = m_host_url[:-1]

        m_subhost = p_url[len(m_host_url):]
        if len(m_subhost) == 1:
            m_subhost = "na"
        return m_host_url, m_subhost

    @staticmethod
    def normalize_slashes(p_url):
        p_url = str(p_url)
        segments = p_url.split('/')
        correct_segments = []
        for segment in segments:
            if segment != '':
                correct_segments.append(segment)
        normalized_url = '/'.join(correct_segments)
        normalized_url = normalized_url.replace("http:/", "http://")
        normalized_url = normalized_url.replace("https:/", "https://")
        normalized_url = normalized_url.replace("ftp:/", "ftp://")
        return normalized_url

    @staticmethod
    def is_url_base_64(p_url):
        if str(p_url).startswith("duplicationHandlerService:"):
            return True
        else:
            return False

    @staticmethod
    def is_uri_validator(p_url):
        try:
            result = urlparse(p_url)
            return all([result.scheme, result.netloc])
        except:
            return False

    @staticmethod
    def clear_folder(p_path):
        for f in os.listdir(p_path):
            try:
                os.remove(os.path.join(p_path, f))
            except Exception as e:
                log.g().e(f"Error removing file {f}: {e}")

    @staticmethod
    def write_content_to_path(p_path, p_content):
        m_url_path = p_path
        file = open(m_url_path, "wb")
        file.write(p_content)
        file.close()

    # Extract URL Host
    @staticmethod
    def get_host_url(p_url):
        m_parsed_uri = urlparse(p_url)
        m_host_url = '{uri.scheme}://{uri.netloc}/'.format(uri=m_parsed_uri)
        if m_host_url.endswith("/"):
            m_host_url = m_host_url[:-1]
        return m_host_url

    @staticmethod
    def clean_text(text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text
