import re
import json
import inspect
from typing import List, Tuple
from collections import defaultdict

import spacy
import iocextract
import ioc_finder
import pycountry
import httpx
from rapidfuzz.fuzz import token_set_ratio
from nltk import TweetTokenizer
from phonenumbers import PhoneNumberMatcher, is_valid_number, region_code_for_number, SUPPORTED_REGIONS
from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from presidio_analyzer.nlp_engine import NlpEngineProvider
from api.nlp_manager.nlp_enums import NLP_REQUEST_COMMANDS


class nlp_controller:

    def __init__(self):
        self.analyzer = self.setup_presidio()
        self.nlp = spacy.load("en_core_web_lg")
        self.EXCLUDED_LABELS = {"TIME", "QUANTITY", "ORDINAL", "MONEY", "DATE", "CARDINAL"}

        self.custom_ioc_patterns = {
            "XMR_WALLET": r"\b4[0-9AB][1-9A-HJ-NP-Za-km-z]{93}\b",
            "API_KEY": r"\bapikey[_-]?[a-zA-Z0-9]{10,}\b",
            "AWS_SECRET": r"\b(?:[A-Za-z0-9/+=]{40})\b",
            "AZURE_RESOURCE_ID": r"/subscriptions/[0-9a-fA-F-]{36}/resourceGroups/[^\s/]+",
            "REGISTRY_KEY": r"HKEY_[A-Z_\\]+",
            "FILE_PATH": r"(?<!://)(?:[a-zA-Z]:)?[\\/](?:[\w\s.-]+[\\/])*[\w\s.-]+\.\w+|(?:^|[^/])/[\w/.-]+",
            "YARA_RULE": r"\brule\s+[\w_]+\b"
        }

        self.country_name_set = {country.name for country in pycountry.countries}

        with open("../../../app/raw/attacks/enterprise-attack.json", "r", encoding="utf-8") as f:
            mitre_data = json.load(f)
            self.mitre_techniques = [
                {
                    "id": ref["external_id"],
                    "name": obj.get("name", ""),
                    "type": obj.get("type", ""),
                    "keywords": f"{ref['external_id']} {obj.get('name', '').lower()}"
                }
                for obj in mitre_data.get("objects", [])
                if obj.get("type") == "attack-pattern"
                for ref in obj.get("external_references", [])
                if ref.get("source_name") == "mitre-attack"
            ]

    @staticmethod
    def setup_presidio():
        configuration = {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_lg"}]
        }
        provider = NlpEngineProvider(nlp_configuration=configuration)
        nlp_engine = provider.create_engine()

        registry = RecognizerRegistry()
        registry.load_predefined_recognizers()

        return AnalyzerEngine(registry=registry, nlp_engine=nlp_engine, supported_languages=["en"])

    @staticmethod
    async def __llama_summarize(text: str, model: str = "tinyllama", summarize: bool = False) -> str:
        text = text[0:500]
        API_URL = "http://168.231.86.34:11434/api/chat"
        model = "llama3.2" if summarize else model
        prompt = (
                "this is data posted on darkweb by a threat actor. Write executive summary only about what is in the report "
                "dont add conclusion or any suggestions. dont add what is not in report. "
                "Start directly with the incident. Do not include any introductions, headings, or phrases like "
                "'Executive summary:', 'Sure', 'Here is the summary:', etc.\n\n" + text
        )

        data = {
            "model": model,
            "messages": [
                {"role": "system",
                 "content": "Treat this prompt as a standalone request. Do not retain or use any previous context."},
                {"role": "user", "content": prompt}
            ],
            "stream": False
        }

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(API_URL, json=data)
                if response.status_code == 200:
                    content = response.json()["message"]["content"].strip()
                    for prefix in [
                        "executive summary:", "summary:", "here is the summary:", "here's the summary:",
                        "sure, here's the summary:", "sure:", "summary -", "summary —"
                    ]:
                        if content.lower().startswith(prefix):
                            return content[len(prefix):].lstrip()
                    return content
                return f"[LLaMA API Error {response.status_code}] {response.text}"
        except Exception as e:
            return f"[LLaMA Exception] {str(e)}"

    def extract_technique_names_from_text(self, text, threshold=70):
        sentences = [s.strip() for s in re.split(r'[.!?\n]', text) if s.strip()]
        matched_names, matched_types = set(), set()

        for sentence in sentences:
            sentence_lower = sentence.lower()
            for t in self.mitre_techniques:
                score = token_set_ratio(sentence_lower, t['keywords'])
                if score >= threshold:
                    matched_names.add(t["name"])
                    matched_types.add(t["type"])

        return sorted(matched_names), sorted(matched_types)

    @staticmethod
    def extract_hashtags_mentions(text: str) -> Tuple[set, set]:
        tokens = TweetTokenizer().tokenize(text)
        return {t for t in tokens if t.startswith('#')}, {t for t in tokens if t.startswith('@')}

    @staticmethod
    def extract_credentials(text: str) -> List[Tuple[str, str]]:
        credentials, email_usernames = [], set()
        text = re.sub(r'[\r\n]+', ' ', text)
        pattern = (
            r'(?i)\b(user[\s_-]*name|user|login|usr)\b\s*[-:=>]*\s*([a-zA-Z0-9._-]{3,30})'
            r'[^\w]{1,15}\b(pass[\s_-]*word|pass|pwd)\b\s*[-:=>]*\s*([^\s;,"\'<>\\]{6,40})'
        )

        for match in re.finditer(pattern, text):
            u, p = match.group(2).strip('.:,;'), match.group(4).strip('.:,;')
            if 3 <= len(u) <= 30 and len(p) >= 6 and re.fullmatch(r'[a-zA-Z0-9._-]{3,30}', u):
                credentials.append((u, p))

        for m in re.findall(r'\b([a-zA-Z0-9._%+-]+)@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b', text):
            local = m[0]
            if 3 <= len(local) <= 30 and re.fullmatch(r'[a-zA-Z0-9._-]+', local):
                email_usernames.add(local)

        return credentials + [(u, "") for u in email_usernames]

    def extract_presidio_entities(self, text, already_found):
        presidio_entities = defaultdict(set)
        EXCLUDED_PRESIDIO_LABELS = {
            "DATE_TIME", "EMAIL_ADDRESS", "IP_ADDRESS",
            "PHONE_NUMBER", "URL", "PERSON", "ORGANIZATION"
        }

        for result in self.analyzer.analyze(text=text, language="en"):
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
                        extracted[name.replace("extract_", "").upper()].add(val)
                except:
                    continue
        try:
            matches = ioc_finder.find_iocs(text)
            for ioc_type, values in matches.items():
                if isinstance(values, dict):
                    for sub_type, sub_values in values.items():
                        extracted[f"{ioc_type}_{sub_type}".upper()].update(sub_values)
                else:
                    extracted[ioc_type.upper()].update(values)
        except:
            pass
        return {label: sorted(values) for label, values in extracted.items()}

    def extract_custom_iocs(self, text):
        custom_iocs = defaultdict(set)
        for label, pattern in self.custom_ioc_patterns.items():
            for match in re.findall(pattern, text):
                custom_iocs[label].add(match.strip())
        return custom_iocs

    @staticmethod
    def extract_phone_data(text, exclude_set):
        detected, countries = set(), set()
        for region_code in SUPPORTED_REGIONS:
            try:
                for match in PhoneNumberMatcher(text, region_code):
                    raw = match.raw_string.strip()
                    num = match.number
                    if is_valid_number(num) and raw not in exclude_set and 10 <= len(re.sub(r'\D', '', raw)) <= 15:
                        detected.add(raw)
                        if raw.startswith('+'):
                            code = region_code_for_number(num)
                            country = pycountry.countries.get(alpha_2=code)
                            if country:
                                countries.add(country.name)
            except:
                continue
        return detected, countries

    def extract_countries_from_text(self, text, from_numbers):
        matches = re.findall(r'\b(' + '|'.join(re.escape(name) for name in self.country_name_set) + r')\b', text,
                             flags=re.IGNORECASE)
        from_text = {pycountry.countries.lookup(name).name for name in matches}
        return from_numbers | from_text

    def extract_spacy_entities(self, text):
        return {(ent.text, ent.label_) for ent in self.nlp(text).ents if ent.label_ not in self.EXCLUDED_LABELS}

    @staticmethod
    def unify_entities(phone_numbers, countries, spacy_entities, iocs, presidio_entities, credentials, hashtags,
                       mentions,
                       mitre_name, mitre_type, summary=None):
        grouped = defaultdict(set)
        for ioc_type, values in iocs.items():
            grouped[ioc_type].update(values)
        grouped["PHONE_NUMBER"].update(phone_numbers)
        grouped["COUNTRY"].update(countries)
        for text, label in spacy_entities:
            grouped[label].add(text)
        for label, values in presidio_entities.items():
            grouped[label].update(values)
        for u, p in credentials:
            grouped["USERNAME"].add(u)
            if p:
                grouped["PASSWORD"].add(p)
        grouped["HASHTAG"].update(hashtags)
        grouped["MENTION"].update(mentions)
        grouped["MITRE_TTP_NAME"].update(mitre_name)
        grouped["MITRE_TTP_TYPE"].update(mitre_type)
        if summary:
            grouped["SUMMARY"].add(summary)
        return {
            f"m_{label.lower().replace(' ', '_')}": sorted(values)
            for label, values in grouped.items()
        }

    @staticmethod
    def clean_text(text):
        return re.sub(r'\s{2,}', '. ', re.sub(r'[\r\n\t\\]+', '. ',
                                              text.encode('utf-8', 'ignore').decode('unicode_escape',
                                                                                    'ignore'))).strip()

    async def __parse(self, text, ai=False):
        text = self.clean_text(text)
        iocs = defaultdict(set, self.extract_iocs_from_text(text))
        for k, v in self.extract_custom_iocs(text).items():
            iocs[k].update(v)
        all_ioc_values = {val for vals in iocs.values() for val in vals}
        phones, countries1 = self.extract_phone_data(text, all_ioc_values)
        countries = self.extract_countries_from_text(text, countries1)
        spacy_ents = self.extract_spacy_entities(text)
        presidio = self.extract_presidio_entities(text, all_ioc_values)
        creds = self.extract_credentials(text)
        tags, mentions = self.extract_hashtags_mentions(text)
        mitre_name, mitre_type = self.extract_technique_names_from_text(text)
        summary = await self.__llama_summarize(text, summarize=True) if ai else None
        grouped = self.unify_entities(phones, countries, spacy_ents, iocs, presidio, creds, tags, mentions, mitre_name,
                                      mitre_type, summary)

        excluded = {"m_percent", "m_iocs", "m_loc", "m_work_of_art", "m_nrp", "m_urls", "m_unencoded_urls"}
        email_keys = {"m_emails", "m_iocs", "m_email_addresses", "m_email_addresses_complete"}
        merged = {}

        for k, values in grouped.items():
            if not values or k in excluded:
                continue

            def valid(is_valid):
                return len(is_valid) <= 15 and re.match(r'^[\w\s]+$', is_valid) if k in {"m_person", "m_org", "m_location"} else True

            if k in email_keys:
                merged.setdefault("m_email", set()).update(v for v in values if v and valid(v))
            else:
                merged.setdefault(k, set()).update(v for v in values if v and valid(v))

        return [{k: v} for k, vs in merged.items() for v in vs]

    async def invoke_trigger(self, command, data=None):
        if command == NLP_REQUEST_COMMANDS.S_PARSE:
            return await self.__parse(data[0][0:1000])
        if command == NLP_REQUEST_COMMANDS.S_PARSE_AI:
            return await self.__parse(data[0][0:1000], True)
        if command == NLP_REQUEST_COMMANDS.S_SUMMARIZE_AI:
            return await self.__llama_summarize(data[0][0:1000], summarize=True)
        return None
