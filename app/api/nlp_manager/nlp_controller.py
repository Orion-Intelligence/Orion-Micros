import spacy
from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from api.nlp_manager.nlp_enums import NLP_REQUEST_COMMANDS


class nlp_controller:

    def __init__(self):
        self.analyzer = AnalyzerEngine()
        btc_pattern = Pattern("BTC Address", r"\b[13][a-km-zA-HJ-NP-Z1-9]{25,34}\b", 0.8)
        btc_recognizer = PatternRecognizer(supported_entity="CRYPTO_BTC", patterns=[btc_pattern])
        self.analyzer.registry.add_recognizer(btc_recognizer)
        self.spacy_nlp = spacy.load("en_core_web_lg")

    def __parse(self, text):
        output = []

        presidio_results = self.analyzer.analyze(text=text, entities=[], language='en')
        for r in presidio_results:
            if r.entity_type != "URL" and r.score >= 0.85:
                entity_text = text[r.start:r.end]
                output.append({
                    r.entity_type: entity_text
                })

        doc = self.spacy_nlp(text)
        for ent in doc.ents:
            output.append({
                ent.label_: ent.text
            })

        return output

    def invoke_trigger(self, command, data=None):
        if command == NLP_REQUEST_COMMANDS.S_PARSE:
            return self.__parse(data[0])
