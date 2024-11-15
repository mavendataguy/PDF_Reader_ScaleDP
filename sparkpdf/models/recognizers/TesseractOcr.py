

from pyspark import keyword_only


from sparkpdf.schemas.Box import Box
from sparkpdf.schemas.Document import Document
from sparkpdf.models.recognizers.BaseOcr import BaseOcr
from sparkpdf.params import *
from ...enums import PSM, OEM, TessLib


class TesseractOcr(BaseOcr):
    """
    Run Tesseract OCR text recognition on images.
    """

    psm = Param(Params._dummy(), "psm",
                           "The desired PageSegMode. Defaults to :attr:`PSM.AUTO",
                           typeConverter=TypeConverters.toInt)

    oem = Param(Params._dummy(), "oem",
                "OCR engine mode. Defaults to :attr:`OEM.DEFAULT`.",
                typeConverter=TypeConverters.toInt)

    tessDataPath = Param(Params._dummy(), "tessDataPath",
                         "Path to tesseract data folder.",
                         typeConverter=TypeConverters.toString)

    
    tessLib = Param(Params._dummy(), "tessLib",
                            "The desired Tesseract library to use. Defaults to :attr:`TESSEROCR`",
                            typeConverter=TypeConverters.toInt)

    defaultParams = {
        "inputCol": "image",
        "outputCol": "text",
        "keepInputData": False,
        "scaleFactor": 1.0,
        "scoreThreshold": 0.5,
        "psm": PSM.AUTO,
        "oem": OEM.DEFAULT,
        "lang": ["eng"],
        "lineTolerance": 0,
        "keepFormatting": False,
        "tessDataPath": "/usr/share/tesseract-ocr/5/tessdata/",
        "tessLib": TessLib.PYTESSERACT,
        "partitionMap": False
    }

    @keyword_only
    def __init__(self, **kwargs):
        super(TesseractOcr, self).__init__()
        self._setDefault(**self.defaultParams)
        self._set(**kwargs)

    def getConfig(self):
        return f"--psm {self.getPsm()} --oem {self.getOem()} -l {self.getLangTess()}"

    def call_pytesseract(self, image, scale_factor, image_path):
        import pytesseract
        res = pytesseract.image_to_data(image, output_type=pytesseract.Output.DATAFRAME, config=self.getConfig())
        res["conf"] = res["conf"] / 100

        if not self.getKeepFormatting():
            res.loc[res["level"] == 4, "conf"] = 1.0
            res["text"] = res["text"].fillna('\n')

        res = res[res["conf"] > self.getScoreThreshold()][['text', 'conf', 'left', 'top', 'width', 'height']]\
            .rename(columns={"conf": "score", "left": "x", "top": "y"})
        res = res[res["text"] != '\n']
        boxes = res.apply(lambda x: Box(*x).toString().scale(1 / scale_factor), axis=1).values.tolist()
        if self.getKeepFormatting():
            text = TesseractOcr.box_to_formatted_text(boxes, self.getLineTolerance())
        else:
            text = " ".join([str(w) for w in res["text"].values.tolist()])

        return Document(path=image_path,
                        text=text,
                        type="text",
                        bboxes=boxes)

    def call_tesserocr(self, image, scale_factor, image_path): # pragma: no cover
        from tesserocr import PyTessBaseAPI, RIL, iterate_level
        
        with PyTessBaseAPI(path=self.getTessDataPath(), psm=self.getPsm(), oem=self.getOem(),
                           lang=self.getLangTess()) as api:
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
                text = TesseractOcr.box_to_formatted_text(boxes, self.getLineTolerance())
            else:
                text = " ".join(texts)

        return Document(path=image_path,
                        text=text,
                        bboxes=boxes,
                        type="text",
                        exception="")

    def call_ocr(self, image, scale_factor, image_path):
        if self.getTessLib() == TessLib.TESSEROCR.value:
            return self.call_tesserocr(image, scale_factor, image_path)
        elif self.getTessLib() == TessLib.PYTESSERACT.value:
            return self.call_pytesseract(image, scale_factor, image_path)
        else:
            raise ValueError(f"Unknown Tesseract library: {self.getTessLib()}")

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
