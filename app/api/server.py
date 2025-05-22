from typing import Dict
from .cti_model.cti_classifier_controller import cti_classifier_controller
from .cti_model.cti_enums import CTI_REQUEST_COMMANDS
from .model.classify_request_model import classify_request_model
from .model.parse_request_model import parse_request_model, parse_cti_model
from .nlp_manager.nlp_controller import nlp_controller
from .nlp_manager.nlp_enums import NLP_REQUEST_COMMANDS
from .ocr_manager.ocr_controller import ocr_controller
from .ocr_manager.ocr_emums import OCR_REQUEST_COMMANDS
from .runtime_parse_manager.runtime_parse_controller import runtime_parse_controller
from .topic_manager.topic_classifier_controller import topic_classifier_controller
from .topic_manager.topic_classifier_enums import TOPIC_CLASSFIER_COMMANDS, TOPIC_CATEGORIES
import asyncio
import logging
from fastapi import FastAPI, Request, HTTPException, UploadFile, File

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class APIService:

    def __init__(self):
        self.app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
        self.m_runtime_parser = runtime_parse_controller()
        self.m_cti_parser = cti_classifier_controller()
        self.semaphore = asyncio.Semaphore(50)
        try:
            self.nlp_controller_instance = nlp_controller()
            self.topic_classifier_instance = topic_classifier_controller()
            self.ocr_controller_instance = ocr_controller()
        except Exception as e:
            logger.critical("Failed to initialize controller instances.", exc_info=True)
            raise RuntimeError("Controller initialization failed") from e

        self.app.add_api_route("/nlp/parse", self.nlp_parse, methods=["POST"])
        self.app.add_api_route("/nlp/parse/ai", self.nlp_parse_ai, methods=["POST"])
        self.app.add_api_route("/nlp/summarize/ai", self.nlp_summarise_ai, methods=["POST"])
        self.app.add_api_route("/cti_classifier/classify", self.cti_classify, methods=["POST"])
        self.app.add_api_route("/topic_classifier/predict", self.topic_classifier_predict, methods=["POST"])
        self.app.add_api_route("/runtime/parse", self.runtime_parse, methods=["POST"])
        self.app.add_api_route("/ocr/parse", self.ocr_parse, methods=["POST"])

    async def cti_classify(self, request: parse_cti_model):
        logger.info("Received request at /cti_classifier/classify")
        async with self.semaphore:
            try:
                result = await asyncio.wait_for(
                    asyncio.to_thread(lambda: self.m_cti_parser.invoke_trigger(CTI_REQUEST_COMMANDS.S_PARSE, [request.data])),
                    timeout=15
                )
                return {"result": result}
            except asyncio.TimeoutError:
                logger.warning("CTI classification request timed out.")
                return {"result": {"label": "none"}}
            except Exception:
                logger.error("Error occurred while processing CTI classification request.", exc_info=True)
                return {"result": {"label": "none"}}

    async def runtime_parse(self, request: Request):
        try:
            payload = await request.json()
            query: Dict[str, str] = payload.get("text", {})

            if not query or all(value == "" for value in query.values()):
                raise HTTPException(status_code=400, detail="No valid query parameters provided or all values are empty")

            response = await asyncio.wait_for(
                self.m_runtime_parser.get_email_username(query),
                timeout=120
            )
            return response

        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="Request timed out while processing")
        except Exception:
            logger.error("Exception occurred during runtime parse", exc_info=True)
            raise HTTPException(status_code=500, detail="An error occurred while processing the request")

    async def process_request(self, request, command, controller, default_result, timeout=60):
        async with self.semaphore:
            try:
                result = await asyncio.wait_for(
                    asyncio.to_thread(lambda: controller(command, request)),
                    timeout=timeout
                )
                return {"result": result}
            except asyncio.TimeoutError:
                logger.warning(f"Request for command '{command}' timed out after {timeout} seconds. Returning default result.")
                return {"result": default_result}
            except Exception:
                logger.error(f"An error occurred while processing request for command '{command}'. Returning default result.", exc_info=True)
                return {"result": default_result}

    async def nlp_parse(self, request: parse_request_model):
        logger.info("Received request at /nlp/parse")
        return await self.process_request(
            request=[request.data],
            command=NLP_REQUEST_COMMANDS.S_PARSE,
            controller=self.nlp_controller_instance.invoke_trigger,
            default_result={}
        )

    async def nlp_parse_ai(self, request: parse_request_model):
        logger.info("Received request at /nlp/parse/ai")
        return await self.process_request(
            request=[request.data],
            command=NLP_REQUEST_COMMANDS.S_PARSE_AI,
            controller=self.nlp_controller_instance.invoke_trigger,
            default_result={}
        )

    async def nlp_summarise_ai(self, request: parse_request_model):
        logger.info("Received request at /nlp/summarize/ai")
        return await self.process_request(
            timeout=60,
            request=[request.data],
            command=NLP_REQUEST_COMMANDS.S_SUMMARIZE_AI,
            controller=self.nlp_controller_instance.invoke_trigger,
            default_result={}
        )

    async def topic_classifier_predict(self, request: classify_request_model):
        logger.info("Received request at /topic_classifier/predict")
        return await self.process_request(
            request=[request.title, request.description, request.keyword],
            command=TOPIC_CLASSFIER_COMMANDS.S_PREDICT_CLASSIFIER,
            controller=self.topic_classifier_instance.invoke_trigger,
            default_result=TOPIC_CATEGORIES.S_THREAD_CATEGORY_GENERAL
        )

    async def ocr_parse(self, file: UploadFile = File(...)):
        logger.info("Received request at /ocr/parse")
        content = await file.read()
        if len(content) > 50 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="File too large. Maximum allowed size is 50MB")

        async with self.semaphore:
            try:
                result = await asyncio.wait_for(
                    asyncio.to_thread(lambda: self.ocr_controller_instance.invoke_trigger(OCR_REQUEST_COMMANDS.S_PARSE, [content])),
                    timeout=60
                )
                return {"text": result}
            except asyncio.TimeoutError:
                logger.warning("OCR parsing timed out.")
                raise HTTPException(status_code=504, detail="OCR request timeout")
            except Exception:
                logger.error("Exception occurred during OCR parsing", exc_info=True)
                raise HTTPException(status_code=500, detail="OCR processing failed")

api_service = APIService()
app = api_service.app
