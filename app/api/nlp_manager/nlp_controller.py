import spacy
import iocextract
import ioc_finder
import inspect
import pycountry
import json, re
import httpx
import requests

from collections import defaultdict
from typing import List, Tuple
from rapidfuzz.fuzz import token_set_ratio
from nltk import TweetTokenizer
from phonenumbers import PhoneNumberMatcher, is_valid_number, region_code_for_number, SUPPORTED_REGIONS
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
from api.nlp_manager.nlp_enums import NLP_REQUEST_COMMANDS

class nlp_controller:


    @staticmethod
    async def __llama_summarize(text: str, model: str = "tinyllama", summarize: bool = False) -> str:
        text = text[0:500]
        API_URL = "http://168.231.86.34:11434/api/chat"

        if summarize:
            model = "llama3.2"
            prompt = (
                "this is data posted on darkweb by a threat actor. Write executive summary only about what is in the report dont add conclusion or any suggestions. dont add what is not in report "
                "Start directly with the incident. Do not include any introductions, headings, or phrases like "
                "'Executive summary:', 'Sure', 'Here is the summary:', etc.\n\n" + text
            )
        else:
            prompt = (
                "Write executive summary only about what is in the report dont add conclusion or any suggestions. dont add what is not in report "
                "Start directly with the incident. Do not include any introductions, headings, or phrases like "
                "'Executive summary:', 'Sure', 'Here is the summary:', etc.\n\n" + text
            )

        data = {
            "model": model,
            "messages": [
                {"role": "system", "content": "Treat this prompt as a standalone request. Do not retain or use any previous context."},
                {"role": "user", "content": prompt}
            ],
            "stream": False
        }

        try:
            async with httpx.AsyncClient(timeout=120) as client:
                response = await client.post(API_URL, json=data)
                if response.status_code == 200:
                    raw_output = response.json()["message"]["content"]
                    clean_output = raw_output.strip()
                    prefixes_to_remove = [
                        "executive summary:", "summary:", "here is the summary:", "here's the summary:",
                        "sure, here's the summary:", "sure:", "summary -", "summary —"
                    ]
                    for prefix in prefixes_to_remove:
                        if clean_output.lower().startswith(prefix):
                            clean_output = clean_output[len(prefix):].lstrip()
                            break
                    return clean_output
                else:
                    return f"[LLaMA API Error {response.status_code}] {response.text}"
        except Exception as e:
            return f"[LLaMA Exception] {str(e)}"

    @staticmethod
    def setup_presidio():
        configuration = {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_lg"}]
        }
        provider = NlpEngineProvider(nlp_configuration=configuration)
        return AnalyzerEngine(nlp_engine=provider.create_engine(), supported_languages=["en"])

    @staticmethod
    def extract_technique_names_from_text(text, threshold=70):
        mitre_path = "../../../app/raw/attacks/enterprise-attack.json"
        with open(mitre_path, "r", encoding="utf-8") as f:
            mitre_data = json.load(f)

        techniques = []
        for obj in mitre_data.get("objects", []):
            if obj.get("type") == "attack-pattern":
                for ref in obj.get("external_references", []):
                    if ref.get("source_name") == "mitre-attack":
                        techniques.append({
                            "id": ref["external_id"],
                            "name": obj.get("name", ""),
                            "type": obj.get("type", "")
                        })

        for t in techniques:
            t['keywords'] = f"{t['id']} {t['name'].lower()}"

        sentences = [s.strip() for s in re.split(r'[.!?\n]', text) if s.strip()]

        matched_names = set()
        matched_types = set()

        for sentence in sentences:
            sentence_lower = sentence.lower()
            for technique in techniques:
                score = token_set_ratio(sentence_lower, technique['keywords'])
                if score >= threshold:
                    matched_names.add(technique["name"])
                    matched_types.add(technique["type"])

        return sorted(matched_names), sorted(matched_types)

    @staticmethod
    def extract_hashtags_mentions(text: str) -> Tuple[set, set]:
        tokenizer = TweetTokenizer()
        tokens = tokenizer.tokenize(text)
        hashtags = {token for token in tokens if token.startswith('#')}
        mentions = {token for token in tokens if token.startswith('@')}
        return hashtags, mentions

    @staticmethod
    def extract_credentials(text: str) -> List[Tuple[str, str]]:
        credentials = []
        text = re.sub(r'[\r\n]+', ' ', text)
        pattern = (
            r'(?i)\b(user[\s_-]*name|user|login|usr)\b\s*[-:=>]*\s*([a-zA-Z0-9._-]{3,30})'
            r'[^\w]{1,15}'
            r'\b(pass[\s_-]*word|pass|pwd)\b\s*[-:=>]*\s*([^\s;,"\'<>\\]{6,40})'
        )
        for match in re.finditer(pattern, text):
            u = match.group(2).strip('.:,;')
            p = match.group(4).strip('.:,;')
            if len(u) < 3 or len(p) < 6: continue
            if re.fullmatch(r'([0-9]{1,3}\.){3}[0-9]{1,3}', u): continue
            if re.fullmatch(r'([0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}', u): continue
            if re.search(r'[:/\\]', u) or u.lower().startswith(('http', 'hxxp', 'ftp')): continue
            if not re.fullmatch(r'[a-zA-Z0-9._-]{3,30}', u): continue
            if '://' in p or p.lower().startswith(('http', 'hxxp', 'ftp')): continue
            if re.fullmatch(r'[0-9a-f]{32,}', p): continue
            if re.search(r'[\[\]{}<>"\'=]', p): continue
            credentials.append((u, p))
        email_usernames = set()
        for match in re.findall(r'\b([a-zA-Z0-9._%+-]+)@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b', text):
            local_part = match[0]
            if 3 <= len(local_part) <= 30 and re.fullmatch(r'[a-zA-Z0-9._-]+', local_part):
                email_usernames.add(local_part)
        for eu in email_usernames:
            credentials.append((eu, ""))
        return credentials

    def extract_presidio_entities(self, text, already_found):
        presidio_entities = defaultdict(set)
        EXCLUDED_PRESIDIO_LABELS = {
            "DATE_TIME", "EMAIL_ADDRESS", "IP_ADDRESS",
            "PHONE_NUMBER", "URL", "PERSON", "ORGANIZATION"
        }

        results = self.analyzer.analyze(text=text, language="en")
        for result in results:
            if result.entity_type in EXCLUDED_PRESIDIO_LABELS:
                continue
            value = text[result.start:result.end]
            if value not in already_found:
                presidio_entities[result.entity_type].add(value)

        return {label: sorted(values) for label, values in presidio_entities.items()}

    @staticmethod
    def extract_iocs_from_text(text):
        extracted = defaultdict(set)
        for name, func in inspect.getmembers(iocextract, inspect.isfunction):
            if name.startswith("extract_"):
                try:
                    results = func(text, refang=True) if 'refang' in inspect.signature(func).parameters else func(text)
                    for val in results:
                        label = name.replace("extract_", "").upper()
                        extracted[label].add(val)
                except:
                    continue
        try:
            matches = ioc_finder.find_iocs(text)
            for ioc_type, values in matches.items():
                if isinstance(values, dict):
                    for sub_type, sub_values in values.items():
                        label = f"{ioc_type}_{sub_type}".upper()
                        extracted[label].update(sub_values)
                else:
                    label = ioc_type.upper()
                    extracted[label].update(values)
        except:
            pass
        return {label: sorted(values) for label, values in extracted.items()}

    @staticmethod
    def extract_custom_iocs(text):
        custom_iocs = defaultdict(set)
        patterns = {
            "XMR_WALLET": r"\b4[0-9AB][1-9A-HJ-NP-Za-km-z]{93}\b",
            "API_KEY": r"\bapikey[_-]?[a-zA-Z0-9]{10,}\b",
            "AWS_SECRET": r"\b(?:[A-Za-z0-9/+=]{40})\b",
            "AZURE_RESOURCE_ID": r"/subscriptions/[0-9a-fA-F-]{36}/resourceGroups/[^\s/]+",
            "REGISTRY_KEY": r"HKEY_[A-Z_\\]+",
            "FILE_PATH": r"(?<!://)(?:[a-zA-Z]:)?[\\/](?:[\w\s.-]+[\\/])*[\w\s.-]+\.\w+|(?:^|[^/])/[\w/.-]+",
            "YARA_RULE": r"\brule\s+[\w_]+\b"
        }
        for label, pattern in patterns.items():
            matches = re.findall(pattern, text)
            for match in matches:
                custom_iocs[label].add(match.strip())
        return custom_iocs

    @staticmethod
    def extract_phone_data(text, exclude_set):
        detected_phone_numbers = set()
        countries_from_numbers = set()
        for region_code in SUPPORTED_REGIONS:
            try:
                phone_matches = PhoneNumberMatcher(text, region_code)
            except:
                continue
            for match in phone_matches:
                raw_number_string = match.raw_string.strip()
                parsed_number = match.number
                if not is_valid_number(parsed_number):
                    continue
                if raw_number_string in exclude_set:
                    continue
                if not (10 <= len(re.sub(r'\D', '', raw_number_string)) <= 15):
                    continue
                detected_phone_numbers.add(raw_number_string)
                if raw_number_string.startswith('+'):
                    try:
                        iso_country_code = region_code_for_number(parsed_number)
                        country = pycountry.countries.get(alpha_2=iso_country_code)
                        if country:
                            countries_from_numbers.add(country.name)
                    except:
                        pass
        return detected_phone_numbers, countries_from_numbers

    @staticmethod
    def extract_countries_from_text(text, countries_from_numbers):
        all_country_names = {country.name for country in pycountry.countries}
        country_pattern = r'\b(' + '|'.join(re.escape(name) for name in all_country_names) + r')\b'
        matched_names = re.findall(country_pattern, text, flags=re.IGNORECASE)
        countries_from_text = {pycountry.countries.lookup(name).name for name in matched_names}
        return countries_from_numbers | countries_from_text

    def extract_spacy_entities(self, text):
        doc = self.nlp(text)
        return {(ent.text, ent.label_) for ent in doc.ents if ent.label_ not in self.EXCLUDED_LABELS}

    @staticmethod
    def unify_entities(phone_numbers, countries, spacy_entities, iocs, presidio_entities, credentials, hashtags, mentions, mitre_name ,mitre_type, summary=None):
        grouped = defaultdict(set)

        for ioc_type, values in iocs.items():
            for val in values:
                grouped[ioc_type].add(val)

        for number in phone_numbers:
            grouped["PHONE_NUMBER"].add(number)

        for country in countries:
            grouped["COUNTRY"].add(country)

        for text, label in spacy_entities:
            grouped[label].add(text)

        for label, values in presidio_entities.items():
            for val in values:
                grouped[label].add(val)

        for user, pwd in credentials:
            grouped["USERNAME"].add(user)
            grouped["PASSWORD"].add(pwd)

        for tag in hashtags:
            grouped["HASHTAG"].add(tag)

        for mention in mentions:
            grouped["MENTION"].add(mention)

        for ttp in mitre_name:
            grouped["MITRE_TTP_NAME"].add(ttp)
        for ttp in mitre_type:
            grouped["MITRE_TTP_TYPE"].add(ttp)

        if summary:
            grouped["SUMMARY"].add(summary)

        normalized_grouped = {
            f"m_{label.lower().replace(' ', '_')}": sorted(values)
            for label, values in grouped.items()
        }
        return normalized_grouped

    @staticmethod
    def clean_text(text):
        text = text.encode('utf-8', 'ignore').decode('unicode_escape', 'ignore')
        text = re.sub(r'[\r\n\t\\]+', '. ', text)
        text = re.sub(r'\s{2,}', '. ', text)
        return text.strip()

    def __parse(self, text, ai=False):
        text = self.clean_text(text)

        iocs = defaultdict(set, self.extract_iocs_from_text(text))
        custom_iocs = self.extract_custom_iocs(text)
        for k, v in custom_iocs.items():
            iocs[k].update(v)

        all_ioc_values = {val for values in iocs.values() for val in values}
        phone_numbers, countries_from_international_numbers = self.extract_phone_data(text, all_ioc_values)
        all_countries = self.extract_countries_from_text(text, countries_from_international_numbers)
        spacy_entities = self.extract_spacy_entities(text)
        presidio_entities = self.extract_presidio_entities(text, already_found=all_ioc_values)
        credentials = self.extract_credentials(text)
        hashtags, mentions = self.extract_hashtags_mentions(text)
        mitre_name, mitre_type = self.extract_technique_names_from_text(text)

        summary = None
        if ai:
            summary = self.__llama_summarize(text)[0:1000]

        grouped_output = self.unify_entities(
            phone_numbers,
            all_countries,
            spacy_entities,
            iocs,
            presidio_entities,
            credentials,
            hashtags,
            mentions,
            mitre_name,
            mitre_type,
            summary=summary
        )

        excluded_keys = {
            "m_percent", "m_iocs", "m_loc", "m_work_of_art", "m_nrp", "m_urls", "m_unencoded_urls"
        }
        email_keys = {
            "m_emails", "m_iocs", "m_email_addresses", "m_email_addresses_complete"
        }

        merged_output = {}
        for k, values in grouped_output.items():
            if not values or k in excluded_keys:
                continue

            def is_valid(v):
                if k in {"m_person", "m_org", "m_location"}:
                    return len(v) <= 15 and re.match(r'^[\w\s]+$', v)
                return True

            if k in email_keys:
                merged_output.setdefault("m_email", set()).update(v for v in values if v and is_valid(v))
            else:
                merged_output.setdefault(k, set()).update(v for v in values if v and is_valid(v))

        formatted_output = [{k: v} for k, vs in merged_output.items() for v in vs]

        return formatted_output

    def invoke_trigger(self, command, data=None):
        if command == NLP_REQUEST_COMMANDS.S_PARSE:
            return self.__parse(data[0])
        if command == NLP_REQUEST_COMMANDS.S_PARSE_AI:
            return self.__parse(data[0], True)
        if command == NLP_REQUEST_COMMANDS.S_SUMMARIZE_AI:
            return self.__llama_summarize(data[0], summarize=True)
