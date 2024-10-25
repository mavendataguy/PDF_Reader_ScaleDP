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
from utils import get_size, cluster


class TesseractOcr(Transformer, HasInputCol, HasOutputCol, HasKeepInputData,
                    DefaultParamsReadable, DefaultParamsWritable, HasScoreThreshold):
    """
    Run Tesseract OCR text recognition on images.
    """
    scaleFactor = Param(Params._dummy(), "scaleFactor",
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
    lang = Param(Params._dummy(), "lang",
                 "Language (e.g., 'eng', 'spa', etc.)",
                 typeConverter=TypeConverters.toString)

    lineTolerance = Param(Params._dummy(), "lineTolerance",
                          "Tolerance for line clustering.",
                          typeConverter=TypeConverters.toInt)

    keepFormatting = Param(Params._dummy(), "keepFormatting",
                           "Whether to keep the original formatting.",
                           typeConverter=TypeConverters.toBoolean)

    @keyword_only

    def __init__(self,
                 inputCol="image",
                 outputCol="text",
                 keepInputData=False,
                 scaleFactor=1.0,
                 scoreThreshold=0.5,
                 psm=PSM.AUTO,
                 oem=OEM.DEFAULT,
                 lang="eng",
                 lineTolerance=0,
                 keepFormatting=False,
                 tessDataPath="/usr/share/tesseract-ocr/5/tessdata/"):
        super(TesseractOcr, self).__init__()
        self._setDefault(inputCol=inputCol,
                         outputCol=outputCol,
                         keepInputData=keepInputData,
                         scaleFactor=scaleFactor,
                         scoreThreshold=scoreThreshold,
                         psm=psm,
                         oem=oem,
                         lang=lang,
                         lineTolerance=lineTolerance,
                         keepFormatting=keepFormatting,
                         tessDataPath=tessDataPath)
    @staticmethod
    def to_formatted_text(lines, character_height):
        output_lines = []
        space_width = TesseractOcr.get_character_width(lines)
        y = 0
        for regions in lines:
            line = ""
            # Add extra empty lines if need
            line_diffs = int((regions[0].y - y) / character_height)
            y = regions[0].y
            if line_diffs > 1:
                for i in range(line_diffs - 1):
                    output_lines.append("")

            prev = 0
            for region in regions:
                # left = region.x - region.width / 2
                # left = int(left / space_width)
                #spaces = max(left - len(line), 1)
                left2 = region.x - prev
                spaces = max(int(left2 / space_width), 1)
                line = line + spaces * " " + region.text
                prev = region.x + region.width
            output_lines.append(line)
        return "\n".join(output_lines)

    @staticmethod
    def get_character_width(lines):
        character_widths = []
        for regions in lines:
            for region in regions:
                width = region.width
                character_widths.append(int(width / len(region.text)))
        return get_size(character_widths)
    def box_to_formatted_text(self, boxes):
        character_height = get_size(boxes, lambda x: x.height)
        line_tolerance = character_height / 3
        if self.getLineTolerance() != 0:
            line_tolerance = self.getLineTolerance()

        lines = cluster(boxes, line_tolerance, key=lambda i: int(i.y))

        lines = [
            sorted(xs, key=lambda i: int(i.x))
            for xs in lines
        ]
        return self.to_formatted_text(lines, character_height)

    def transform_udf(self, image):
        logging.info("Run Tesseract OCR")
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
            with PyTessBaseAPI(path=self.getTessDataPath(), psm=self.getPsm(), oem=self.getOem(),
                               lang=self.getLang()) as api:
                api.SetVariable("debug_file", "ocr.log")
                api.SetImage(tn_image)
                api.SetVariable("save_blob_choices", "T")
                api.Recognize()
                iterator = api.GetIterator()
                boxes = []
                texts = []

                level = RIL.WORD
                for r in iterate_level(iterator, level):
                    conf = r.Confidence(level) / 100
                    text = r.GetUTF8Text(level)
                    box = r.BoundingBox(level)
                    if conf > self.getScoreThreshold():
                        boxes.append(Box(text, conf, box[0], box[1], abs(box[2] - box[0]), abs(box[3] - box[1])).scale(1 / factor))
                        texts.append(text)
                if self.getKeepFormatting():
                    text = self.box_to_formatted_text(boxes)
                else:
                    text = " ".join(texts)
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
                             text=text,
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

    def setLang(self, value):
        """
        Sets the value of :py:attr:`lang`.
        """
        return self._set(lang=value)

    def getLang(self):
        """
        Sets the value of :py:attr:`lang`.
        """
        return self.getOrDefault(self.lang)

    def setLineTolerance(self, value):
        """
        Sets the value of :py:attr:`lineTolerance`.
        """
        return self._set(lineTolerance=value)

    def getLineTolerance(self):
        """
        Gets the value of :py:attr:`lineTolerance`.
        """
        return self.getOrDefault(self.lineTolerance)

    def setKeepFormatting(self, value):
        """
        Sets the value of :py:attr:`keepFormatting`.
        """
        return self._set(keepFormatting=value)

    def getKeepFormatting(self):
        """
        Gets the value of :py:attr:`keepFormatting`.
        """
        return self.getOrDefault(self.keepFormatting)

