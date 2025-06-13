from typing import Dict
from fastapi import FastAPI, Request, HTTPException, UploadFile, File
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
import concurrent.futures
from contextlib import asynccontextmanager

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class APIService:
    MAX_CONCURRENT_REQUESTS = 20

    def __init__(self):
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.MAX_CONCURRENT_REQUESTS)
        loop = asyncio.get_event_loop()
        loop.set_default_executor(executor)

        self.app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
        self.semaphore = asyncio.Semaphore(self.MAX_CONCURRENT_REQUESTS)
        self.waiting_requests = 0
        self.active_requests = 0
        self.waiting_lock = asyncio.Lock()

        self.m_runtime_parser = runtime_parse_controller()
        self.m_cti_parser = cti_classifier_controller()

        try:
            self.nlp_controller_instance = nlp_controller()
            self.topic_classifier_instance = topic_classifier_controller()
            self.ocr_controller_instance = ocr_controller()
        except Exception:
            logger.critical("Failed to initialize controller instances.", exc_info=True)
            raise RuntimeError("Controller initialization failed")

        self.app.add_api_route("/nlp/parse", self.nlp_parse, methods=["POST"])
        self.app.add_api_route("/nlp/parse/ai", self.nlp_parse_ai, methods=["POST"])
        self.app.add_api_route("/nlp/summarize/ai", self.nlp_summarise_ai, methods=["POST"])
        self.app.add_api_route("/cti_classifier/classify", self.cti_classify, methods=["POST"])
        self.app.add_api_route("/topic_classifier/predict", self.topic_classifier_predict, methods=["POST"])
        self.app.add_api_route("/runtime/parse", self.runtime_parse, methods=["POST"])
        self.app.add_api_route("/ocr/parse", self.ocr_parse, methods=["POST"])
        self.app.add_api_route("/debug/semaphore_status", self.semaphore_status, methods=["GET"])

        loop.create_task(self.log_queue_size())

    @asynccontextmanager
    async def track_waiting(self, endpoint: str = "unknown"):
        async with self.waiting_lock:
            self.waiting_requests += 1
            logger.info(f"[START_WAIT] Request to {endpoint}. Waiting: {self.waiting_requests}")

        try:
            async with self.semaphore:
                async with self.waiting_lock:
                    self.waiting_requests -= 1
                    self.active_requests += 1
                    logger.info(f"[START] Request to {endpoint}. In Use: {self.active_requests}")
                yield
        finally:
            async with self.waiting_lock:
                self.active_requests -= 1
                logger.info(f"[END] Request to {endpoint}. In Use: {self.active_requests}")
                logger.info(
                    f"[CLOSED] Request to {endpoint}. Final state — Waiting: {self.waiting_requests}, In Use: {self.active_requests}")

    async def semaphore_status(self):
        async with self.waiting_lock:
            return {
                "waiting_queue_size": self.waiting_requests,
                "active_slots": self.active_requests,
                "available_slots": max(0, self.MAX_CONCURRENT_REQUESTS - self.active_requests)
            }

    async def log_queue_size(self):
        while True:
            async with self.waiting_lock:
                logger.info(
                    f"[Queue Monitor] Waiting: {self.waiting_requests}, "
                    f"In Use: {self.active_requests}, "
                    f"Available: {max(0, self.MAX_CONCURRENT_REQUESTS - self.active_requests)}"
                )
            await asyncio.sleep(5)

    async def cti_classify(self, request: parse_cti_model):
        logger.info("Received request at /cti_classifier/classify")
        async with self.track_waiting("/cti_classifier/classify"):
            try:
                result = await asyncio.wait_for(
                    self.m_cti_parser.invoke_trigger(CTI_REQUEST_COMMANDS.S_PARSE, [request.data]),
                    timeout=15
                )
                return {"result": result}
            except asyncio.TimeoutError:
                logger.warning("CTI classification request timed out.")
                return {"result": {"label": "none"}}
            except Exception:
                logger.error("Error in CTI classification", exc_info=True)
                return {"result": {"label": "none"}}

    async def runtime_parse(self, request: Request):
        async with self.track_waiting("/runtime/parse"):
            try:
                payload = await request.json()
                query: Dict[str, str] = payload.get("text", {})
                if not query or all(value == "" for value in query.values()):
                    raise HTTPException(status_code=400, detail="Invalid query")

                response = await asyncio.wait_for(
                    self.m_runtime_parser.get_email_username(query),
                    timeout=120
                )
                return response
            except asyncio.TimeoutError:
                raise HTTPException(status_code=504, detail="Request timeout")
            except Exception:
                logger.error("Error in runtime parse", exc_info=True)
                raise HTTPException(status_code=500, detail="Server error")

    async def process_request(self, request, command, controller, default_result, timeout=60, endpoint="unknown"):
        async with self.track_waiting(endpoint):
            try:
                result = await asyncio.wait_for(
                    controller(command, request),
                    timeout=timeout
                )
                return {"result": result}
            except asyncio.TimeoutError:
                logger.warning(f"Timeout on {endpoint}")
                return {"result": default_result}
            except Exception:
                logger.error(f"Error in {endpoint}", exc_info=True)
                return {"result": default_result}

    async def nlp_parse(self, request: parse_request_model):
        logger.info("Received request at /nlp/parse")
        return await self.process_request(
            request=[request.data],
            command=NLP_REQUEST_COMMANDS.S_PARSE,
            controller=self.nlp_controller_instance.invoke_trigger,
            default_result={},
            endpoint="/nlp/parse"
        )

    async def nlp_parse_ai(self, request: parse_request_model):
        logger.info("Received request at /nlp/parse/ai")
        return await self.process_request(
            request=[request.data],
            command=NLP_REQUEST_COMMANDS.S_PARSE_AI,
            controller=self.nlp_controller_instance.invoke_trigger,
            default_result={},
            endpoint="/nlp/parse/ai"
        )

    async def nlp_summarise_ai(self, request: parse_request_model):
        logger.info("Received request at /nlp/summarize/ai")
        async with self.track_waiting("/nlp/summarize/ai"):
            try:
                result = await asyncio.wait_for(
                    self.nlp_controller_instance.invoke_trigger(
                        NLP_REQUEST_COMMANDS.S_SUMMARIZE_AI, [request.data]
                    ),
                    timeout=100
                )
                return {"result": result}
            except asyncio.TimeoutError:
                logger.warning("Summarization timeout")
                return {"result": {}}
            except Exception:
                logger.error("Summarization failed", exc_info=True)
                return {"result": {}}

    async def topic_classifier_predict(self, request: classify_request_model):
        logger.info("Received request at /topic_classifier/predict")
        return await self.process_request(
            request=[request.title, request.description, request.keyword],
            command=TOPIC_CLASSFIER_COMMANDS.S_PREDICT_CLASSIFIER,
            controller=self.topic_classifier_instance.invoke_trigger,
            default_result=[TOPIC_CATEGORIES.S_THREAD_CATEGORY_GENERAL],
            endpoint="/topic_classifier/predict"
        )

    async def ocr_parse(self, file: UploadFile = File(...)):
        logger.info("Received request at /ocr/parse")
        content = await file.read()
        if len(content) > 50 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="File too large")

        async with self.track_waiting("/ocr/parse"):
            try:
                result = await asyncio.wait_for(
                    self.ocr_controller_instance.invoke_trigger(OCR_REQUEST_COMMANDS.S_PARSE, [content]),
                    timeout=60
                )
                return {"text": result}
            except asyncio.TimeoutError:
                logger.warning("OCR timeout")
                raise HTTPException(status_code=504, detail="Timeout")
            except Exception:
                logger.error("OCR error", exc_info=True)
                raise HTTPException(status_code=500, detail="Error processing file")


api_service = APIService()
app = api_service.app
