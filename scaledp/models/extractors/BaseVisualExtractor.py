import json
import logging
import traceback
from pyspark.sql.types import *
from pyspark.ml import Transformer
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql.functions import lit, udf

from scaledp.params import *
from scaledp.schemas.Image import Image
from scaledp.schemas.ExtractorOutput import ExtractorOutput


class VisualExtractorError(Exception):
    pass


class BaseVisualExtractor(
    Transformer,
    HasInputCol,
    HasOutputCol,
    HasKeepInputData,
    HasPathCol,
    DefaultParamsReadable,
    DefaultParamsWritable,
    HasNumPartitions,
    HasPageCol,
    HasColumnValidator,
    HasDefaultEnum,
    HasPropagateError,
):

    def get_params(self):
        return json.dumps({k.name: v for k, v in self.extractParamMap().items()})

    @classmethod
    def call_extractor(cls, images, params):
        raise NotImplementedError("Subclasses should implement this method")

    def transform_udf(self, image, params=None):
        logging.info("Run Image Data Extractor")
        if params is None:
            params = self.get_params()
        params = json.loads(params)
        if not isinstance(image, Image):
            image = Image(**image.asDict())
        if image.exception != "":
            return ExtractorOutput(
                path=image.path,
                data="",
                type="extractor",
                exception=image.exception,
            )
        try:

            result = self.call_extractor([image], params)
        except Exception as e:
            exception = traceback.format_exc()
            exception = f"{self.uid}: Error in data extraction: {exception}, {image.exception}"
            logging.warning(f"{self.uid}: Error in data extraction.")
            if self.getPropagateError():
                raise VisualExtractorError() from e
            return ExtractorOutput(
                path=image.path, data="", type="detector", exception=exception
            )
        return result[0]

    def _transform(self, dataset):
        out_col = self.getOutputCol()
        in_col = self._validate(self.getInputCol(), dataset)

        result = dataset.withColumn(
            out_col,
            udf(self.transform_udf, ExtractorOutput.get_schema())(in_col, lit(self.get_params())),
        )
        if not self.getKeepInputData():
            result = result.drop(in_col)
        return result
