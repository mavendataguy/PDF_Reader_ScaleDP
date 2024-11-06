import traceback
import logging

from pyspark import keyword_only
from pyspark.sql.functions import udf
from pyspark.ml import Transformer
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable

from sparkpdf.schemas.Box import Box
from sparkpdf.schemas.Image import Image
from sparkpdf.schemas.Document import Document
from sparkpdf.params import *
from ...enums import PSM, OEM, TessLib
from ...utils import get_size, cluster


class TesseractOcr(Transformer, HasInputCol, HasOutputCol, HasKeepInputData, HasDefaultEnum,
                    DefaultParamsReadable, DefaultParamsWritable, HasScoreThreshold, HasColumnValidator):
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
    
    tessLib = Param(Params._dummy(), "tessLib",
                            "The desired Tesseract library to use. Defaults to :attr:`TESSEROCR`",
                            typeConverter=TypeConverters.toInt)

    @keyword_only
    def __init__(self,
                 inputCol="image",
                 outputCol="text",
                 keepInputData=False,
                 scaleFactor=1.0,
                 scoreThreshold=0.5,
                 psm=PSM.AUTO.value,
                 oem=OEM.DEFAULT.value,
                 lang="eng",
                 lineTolerance=0,
                 keepFormatting=False,
                 tessDataPath="/usr/share/tesseract-ocr/5/tessdata/",
                 tessLib=TessLib.PYTESSERACT.value):
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
                         tessDataPath=tessDataPath,
                         tessLib=tessLib)

    @staticmethod
    def to_formatted_text(lines, character_height):
        output_lines = []
        space_width = TesseractOcr.get_character_width(lines)
        y = 0
        for regions in lines:
            line = ""
            # Add extra empty lines if need
            line_diffs = int((regions[0].y - y) / (character_height * 2))
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

    def getConfig(self):
        return f"--psm {self.getPsm()} --oem {self.getOem()} -l {self.getLang()}"

    def call_pytesseract(self, image, scale_factor, image_path):
        import pytesseract
        res = pytesseract.image_to_data(image, output_type=pytesseract.Output.DATAFRAME, config=self.getConfig())
        res["conf"] = res["conf"] / 100

        if not self.getKeepFormatting():
            res.loc[res["level"] == 4, "conf"] = 1.0
            res["text"] = res["text"].fillna('\n')

        res = res[res["conf"] > self.getScoreThreshold()][['text', 'conf', 'left', 'top', 'width', 'height']]\
            .rename(columns={"conf": "score", "left": "x", "top": "y"})
        boxes = res.apply(lambda x: Box(*x).toString().scale(1 / scale_factor), axis=1).values.tolist()
        if self.getKeepFormatting():
            text = self.box_to_formatted_text(boxes)
        else:
            text = " ".join([str(w) for w in res["text"].values.tolist()])
        return Document(path=image_path,
                        text=text,
                        type="text",
                        bboxes=boxes)

    def call_tesserocr(self, image, scale_factor, image_path): # pragma: no cover
        from tesserocr import PyTessBaseAPI, RIL, iterate_level
        
        with PyTessBaseAPI(path=self.getTessDataPath(), psm=self.getPsm(), oem=self.getOem(),
                           lang=self.getLang()) as api:
            api.SetVariable("debug_file", "ocr.log")
            api.SetImage(image)
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
                    boxes.append(
                        Box(text, conf, box[0], box[1], abs(box[2] - box[0]), abs(box[3] - box[1])).scale(1 / scale_factor))
                    texts.append(text)
            if self.getKeepFormatting():
                text = self.box_to_formatted_text(boxes)
            else:
                text = " ".join(texts)

        return Document(path=image_path,
                        text=text,
                        bboxes=boxes,
                        type="text",
                        exception="")

    def transform_udf(self, image):
        logging.info("Run Tesseract OCR")
        if not isinstance(image, Image):
            image = Image(**image.asDict())
        if image.exception != "":
            return Document(path=image.path,
                            text="",
                            bboxes=[],
                            type="text",
                            exception=image.exception)
        try:
            image_pil = image.to_pil()
            scale_factor = self.getScaleFactor()
            if scale_factor != 1.0:
                resized_image = image_pil.resize((int(image_pil.width * scale_factor), int(image_pil.height * scale_factor)))
            else:
                resized_image = image_pil
            if self.getTessLib() == TessLib.TESSEROCR.value:
                result = self.call_tesserocr(resized_image, scale_factor, image.path)
            elif self.getTessLib() == TessLib.PYTESSERACT.value:
                result = self.call_pytesseract(resized_image, scale_factor, image.path)
            else:
                raise ValueError(f"Unknown Tesseract library: {self.getTessLib()}")
        except Exception as e:
            exception = traceback.format_exc()
            exception = f"{self.uid}: Error in text recognition: {exception}, {image.exception}"
            logging.warning(f"{self.uid}: Error in text recognition.")
            return Document(path=image.path,
                            text="",
                            bboxes=[],
                            type="ocr",
                            exception=exception)
        return result

    def _transform(self, dataset):
        out_col = self.getOutputCol()
        input_col = self._validate(self.getInputCol(), dataset)
        result = dataset.withColumn(out_col, udf(self.transform_udf, Document.get_schema())(input_col))
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
