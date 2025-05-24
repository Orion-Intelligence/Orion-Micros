import asyncio
from api.topic_manager.topic_classifier_enums import TOPIC_CLASSFIER_MODEL, TOPIC_CLASSFIER_COMMANDS
from api.topic_manager.topic_classifier_model import topic_classifier_model

class topic_classifier_controller:

    def __init__(self):
        self.__m_classifier = topic_classifier_model()

    async def __predict_classifier(self, p_title, p_description, p_keyword):
        return await asyncio.to_thread(
            self.__m_classifier.sync_invoke_trigger,
            TOPIC_CLASSFIER_MODEL.S_PREDICT_CLASSIFIER,
            [p_title, p_description, p_keyword]
        )

    async def __clean_classifier(self):
        await asyncio.to_thread(
            self.__m_classifier.sync_invoke_trigger,
            TOPIC_CLASSFIER_MODEL.S_CLEAN_CLASSIFIER
        )
        self.__m_classifier = None

    async def invoke_trigger(self, p_command, p_data=None):
        if p_command == TOPIC_CLASSFIER_COMMANDS.S_PREDICT_CLASSIFIER:
            return await self.__predict_classifier(p_data[0], p_data[1], p_data[2])
        if p_command == TOPIC_CLASSFIER_COMMANDS.S_CLEAN_CLASSIFIER:
            return await self.__clean_classifier()
