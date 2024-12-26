import traceback
import logging
import fitz
from pyspark import keyword_only
from pyspark.sql.functions import udf, posexplode_outer
from pyspark.sql.types import ArrayType

from scaledp.schemas.Image import Image
from pyspark.ml import Transformer
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from scaledp.params import *
from scaledp.enums import ImageType


class PdfDataToImage(
    Transformer,
    HasInputCol,
    HasOutputCol,
    HasKeepInputData,
    HasImageType,
    HasPathCol,
    HasResolution,
    HasPageCol,
    DefaultParamsReadable,
    DefaultParamsWritable,
    HasColumnValidator,
    HasDefaultEnum,
):
    """
    Extract image from PDF file
    """

    pageLimit = Param(
        Params._dummy(),
        "pageLimit",
        "Limit number of pages to convert to image",
        typeConverter=TypeConverters.toInt,
    )

    defaultParams = {
        "inputCol": "content",
        "outputCol": "image",
        "pathCol": "path",
        "pageCol": "page",
        "keepInputData": False,
        "imageType": ImageType.FILE,
        "resolution": 300,
        "pageLimit": 0,
    }

    @keyword_only
    def __init__(self, **kwargs):
        super(PdfDataToImage, self).__init__()
        self._setDefault(**self.defaultParams)
        self._set(**kwargs)

    def transform_udf(self, input, path):
        logging.info("Run Pdf Data to Image")
        try:
            doc = fitz.open("pdf", input)
            if len(doc) == 0:
                raise ValueError("Empty PDF document.")
            if self.getPageLimit():
                doc = doc[: self.getPageLimit()]
            for page in doc:
                pix = page.get_pixmap(
                    matrix=fitz.Identity,
                    dpi=self.getResolution(),
                    colorspace=fitz.csRGB,
                    clip=None,
                    alpha=False,
                    annots=True,
                )
                yield Image.from_binary(
                    pix.pil_tobytes("png"),
                    path,
                    self.getImageType(),
                    resolution=self.getResolution(),
                    width=pix.width,
                    height=pix.height,
                )

        except Exception:
            exception = traceback.format_exc()
            exception = (
                f"{self.uid}: Error during extract image from the PDF document: {exception}"
            )
            logging.warning(exception)
            return [Image(path=path, exception=exception)]

    def _transform(self, dataset):
        out_col = self.getOutputCol()
        input_col = self._validate(self.getInputCol(), dataset)
        path_col = dataset[self.getPathCol()]

        selCol = dataset.columns + [
            posexplode_outer(
                udf(self.transform_udf, ArrayType(Image.get_schema()))(input_col, path_col)
            ).alias(self.getPageCol(), out_col),
        ]

        result = dataset.select(selCol)
        if not self.getKeepInputData():
            result = result.drop(input_col)
        return result

    def getPageLimit(self):
        return self.getOrDefault(self.pageLimit)

    def setPageLimit(self, value):
        return self._set(pageNumber=value)
