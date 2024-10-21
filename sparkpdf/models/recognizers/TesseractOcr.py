import traceback
import logging

from tesserocr import PyTessBaseAPI, PSM, OEM, RIL, iterate_level
from pyspark import keyword_only
from pyspark.sql.functions import udf
from pyspark.ml import Transformer
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable

from sparkpdf.schemas.Box import Box
from sparkpdf.schemas.Image import Image
from sparkpdf.schemas.OcrOutput import OcrOutput
from sparkpdf.params import *


class TesseractOcr(Transformer, HasInputCol, HasOutputCol, HasKeepInputData,
                    DefaultParamsReadable, DefaultParamsWritable):
    """
    Run Tesseract OCR text recognition on images.
    """
    scaleFactor = Param(Params._dummy(), "scaleFactor",
                      "Scale Factor.",
                      typeConverter=TypeConverters.toFloat)

    scoreThreshold = Param(Params._dummy(), "scoreThreshold",
                        "Scale Factor.",
                        typeConverter=TypeConverters.toFloat)

    psm = Param(Params._dummy(), "psm",
                           "The desired PageSegMode. Defaults to :attr:`PSM.AUTO",
                           typeConverter=TypeConverters.toInt)

    oem = Param(Params._dummy(), "oem",
                "OCR engine mode. Defaults to :attr:`OEM.DEFAULT`.",
                typeConverter=TypeConverters.toInt)

    tessDataPath = Param(Params._dummy(), "tessDataPath",
                         "Path to tesseract data folder.",
                         typeConverter=TypeConverters.toString)


    @keyword_only
    def __init__(self):
        super(TesseractOcr, self).__init__()
        self._setDefault(outputCol='text')
        self._setDefault(inputCol='image')
        self._setDefault(keepInputData=False)
        self._setDefault(scaleFactor=1.0)
        self._setDefault(scoreThreshold=0.5)
        self._setDefault(psm=PSM.AUTO)
        self._setDefault(oem=OEM.DEFAULT)
        self._setDefault(tessDataPath="/usr/share/tesseract-ocr/5/tessdata/")

    def transform_udf(self, image):
        try:
            if not isinstance(image, Image):
                image = Image(**image.asDict())
            if image.exception != "":
                return OcrOutput(path=image.path,
                             text="",
                             bboxes=[],
                             type="text",
                             exception=image.exception)
            image_pil = image.to_pil()
            factor = self.getScaleFactor()
            if factor != 1.0:
                tn_image = image_pil.resize((int(image_pil.width * factor), int(image_pil.height * factor)))
            else:
                tn_image = image_pil
            with PyTessBaseAPI(path=self.getTessDataPath(), psm=self.getPsm(), oem=self.getOem()) as api:
                api.SetVariable("debug_file", "ocr.log")
                api.SetImage(tn_image)
                api.SetVariable("save_blob_choices", "T")
                api.Recognize()
                iterator = api.GetIterator()
                boxes = []
                texts = []

                level = RIL.WORD
                for r in iterate_level(iterator, level):
                    conf = r.Confidence(level)
                    text = r.GetUTF8Text(level)
                    box = r.BoundingBox(level)
                    boxes.append(Box(text, conf, box[0], box[1], abs(box[2]-box[0]), abs(box[3]-box[1])).scale(1/factor))
                    texts.append(text)
        except Exception as e:
            exception = traceback.format_exc()
            exception = f"{self.uid}: Error in text recognition: {exception}, {image.exception}"
            logging.warning(f"{self.uid}: Error in text recognition.")
            return OcrOutput(path=image.path,
                             text="",
                             bboxes=[],
                             type="text",
                             exception=exception)

        return OcrOutput(path=image.path,
                             text=" ".join(texts),
                             bboxes=boxes,
                             type="text",
                             exception="")

    def _transform(self, dataset):
        out_col = self.getOutputCol()
        if self.getInputCol() not in dataset.columns:
            input_col = self.getInputCol()
            raise ValueError(f"Missing input column in transformer {self.uid}: Column '{input_col}' is not present.")
        input_col = dataset[self.getInputCol()]
        result = dataset.withColumn(out_col, udf(self.transform_udf, OcrOutput.get_schema())(input_col))
        if not self.getKeepInputData():
            result = result.drop(input_col)
        return result

    def setScaleFactor(self, value):
        """
        Sets the value of :py:attr:`scaleFactor`.
        """
        return self._set(scaleFactor=value)

    def getScaleFactor(self):
        """
        Sets the value of :py:attr:`scaleFactor`.
        """
        return self.getOrDefault(self.scaleFactor)

    def setScoreThreshold(self, value):
        """
        Sets the value of :py:attr:`scoreThreshold`.
        """
        return self._set(scoreThreshold=value)

    def getScoreThreshold(self):
        """
        Sets the value of :py:attr:`scoreThreshold`.
        """
        return self.getOrDefault(self.scoreThreshold)

    def setPsm(self, value):
        """
        Sets the value of :py:attr:`psm`.
        """
        return self._set(psm=value)

    def getPsm(self):
        """
        Sets the value of :py:attr:`psm`.
        """
        return self.getOrDefault(self.psm)

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
