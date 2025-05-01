from PIL import Image
from pdf2image import convert_from_bytes
import pytesseract
import io
import filetype

from api.ocr_manager.ocr_emums import OCR_REQUEST_COMMANDS


class ocr_controller:

    def __init__(self):
        pass

    @staticmethod
    def __parse(file_bytes: bytes):
        kind = filetype.guess(file_bytes)
        if not kind:
            return ""

        if kind.mime == 'application/pdf':
            images = convert_from_bytes(file_bytes, dpi=150)
            text = []
            for img in images:
                text.append(pytesseract.image_to_string(img))
            return "\n".join(text)

        elif kind.mime.startswith('image/'):
            img = Image.open(io.BytesIO(file_bytes))
            text = pytesseract.image_to_string(img)
            return text

        return ""

    def invoke_trigger(self, command, data=None):
        if command == OCR_REQUEST_COMMANDS.S_INIT:
            return True
        if command == OCR_REQUEST_COMMANDS.S_PARSE:
            return self.__parse(data[0])
