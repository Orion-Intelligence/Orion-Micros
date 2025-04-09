from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax
from api.cti_model.cti_enums import CTI_REQUEST_COMMANDS
import torch


class cti_classifier_controller:

    def __init__(self):
        self.local_model_dir = "./raw/model/cti_classifier"

        self.tokenizer = AutoTokenizer.from_pretrained(self.local_model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.local_model_dir)

        self.label_map = {
            0: "normal",
            1: "cti"
        }

    def __parse(self, data):
        inputs = self.tokenizer(data, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = softmax(outputs.logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()

        return {
            "label": self.label_map[pred]
        }

    def invoke_trigger(self, command, data=None):
        if command == CTI_REQUEST_COMMANDS.S_PARSE:
            return self.__parse(data[0])
