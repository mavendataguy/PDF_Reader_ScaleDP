from types import MappingProxyType
from typing import Any

from pyspark import keyword_only
from pyspark.ml.param import Param, Params, TypeConverters

from scaledp.params import CODE_TO_LANGUAGE, LANGUAGE_TO_TESSERACT_CODE
from scaledp.schemas.Box import Box
from scaledp.schemas.Document import Document

from ...enums import OEM, PSM, TessLib
from .BaseRecognizer import BaseRecognizer


class TesseractRecognizer(BaseRecognizer):
    """
    Run Tesseract text recognition on images.
    """

    oem = Param(
        Params._dummy(),
        "oem",
        "OCR engine mode. Defaults to :attr:`OEM.DEFAULT`.",
        typeConverter=TypeConverters.toInt,
    )

    tessDataPath = Param(
        Params._dummy(),
        "tessDataPath",
        "Path to tesseract data folder.",
        typeConverter=TypeConverters.toString,
    )

    tessLib = Param(
        Params._dummy(),
        "tessLib",
        "The desired Tesseract library to use. Defaults to :attr:`TESSEROCR`",
        typeConverter=TypeConverters.toInt,
    )

    defaultParams = MappingProxyType(
        {
            "inputCols": ["image", "boxes"],
            "outputCol": "text",
            "keepInputData": False,
            "scaleFactor": 1.0,
            "scoreThreshold": 0.5,
            "oem": OEM.DEFAULT,
            "lang": ["eng"],
            "lineTolerance": 0,
            "keepFormatting": False,
            "tessDataPath": "/usr/share/tesseract-ocr/5/tessdata/",
            "tessLib": TessLib.PYTESSERACT,
            "partitionMap": False,
            "numPartitions": 0,
            "pageCol": "page",
            "pathCol": "path",
        },
    )

    @keyword_only
    def __init__(self, **kwargs: Any) -> None:
        super(TesseractRecognizer, self).__init__()
        self._setDefault(**self.defaultParams)
        self._set(**kwargs)

    @classmethod
    def call_pytesseract(cls, images, boxes, params):
        raise NotImplementedError("Pytesseract version not implemented yet.")

    @staticmethod
    def getLangTess(params):
        return "+".join(
            [
                LANGUAGE_TO_TESSERACT_CODE[CODE_TO_LANGUAGE[lang]]
                for lang in params["lang"]
            ],
        )

    @classmethod
    def call_tesserocr(cls, images, detected_boxes, params):  # pragma: no cover
        from tesserocr import PyTessBaseAPI

        results = []
        lang = cls.getLangTess(params)
        with PyTessBaseAPI(
            path=params["tessDataPath"],
            psm=PSM.SINGLE_WORD,
            oem=params["oem"],
            lang=lang,
        ) as api:
            api.SetVariable("debug_file", "ocr.log")

            for (image, image_path), detected_box in zip(images, detected_boxes):
                api.SetImage(image)

                boxes = []
                texts = []

                for b in detected_box.bboxes:
                    box = b
                    if isinstance(box, dict):
                        box = Box(**box)
                    if not isinstance(box, Box):
                        box = Box(**box.asDict())
                    scaled_box = box.scale(params["scaleFactor"], padding=0)
                    api.SetRectangle(
                        scaled_box.x,
                        scaled_box.y,
                        scaled_box.width,
                        scaled_box.height,
                    )
                    box.text = api.GetUTF8Text().replace("\n", "")
                    box.score = api.MeanTextConf() / 100
                    if box.score > params["scoreThreshold"]:
                        boxes.append(box)
                        texts.append(box.text)
                if params["keepFormatting"]:
                    text = TesseractRecognizer.box_to_formatted_text(
                        boxes,
                        params["lineTolerance"],
                    )
                else:
                    text = " ".join(texts)

                results.append(
                    Document(
                        path=image_path,
                        text=text,
                        bboxes=boxes,
                        type="text",
                        exception="",
                    ),
                )
        return results

    @classmethod
    def call_recognizer(cls, images, boxes, params):
        if params["tessLib"] == TessLib.TESSEROCR.value:
            return cls.call_tesserocr(images, boxes, params)
        if params["tessLib"] == TessLib.PYTESSERACT.value:
            return cls.call_pytesseract(images, boxes, params)
        raise ValueError(f"Unknown Tesseract library: {params['tessLib']}")

    def setOem(self, value):
        """
        Sets the value of :py:attr:`oem`.
        """
        return self._set(oem=value)

    def getOem(self):
        """
        Sets the value of :py:attr:`oem`.
        """
        return self.getOrDefault(self.oem)

    def setTessDataPath(self, value):
        """
        Sets the value of :py:attr:`tessDataPath`.
        """
        return self._set(tessDataPath=value)

    def getTessDataPath(self):
        """
        Sets the value of :py:attr:`tessDataPath`.
        """
        return self.getOrDefault(self.tessDataPath)

    def setTessLib(self, value):
        """
        Sets the value of :py:attr:`tessLib`.
        """
        return self._set(tessLib=value)

    def getTessLib(self):
        """
        Gets the value of :py:attr:`tessLib`.
        """
        return self.getOrDefault(self.tessLib)
