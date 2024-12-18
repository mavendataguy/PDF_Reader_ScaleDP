import json
import traceback
import logging

from pyspark.sql.functions import udf, pandas_udf, lit
from pyspark.sql.types import ArrayType

from scaledp.params import *
from pyspark.ml import Transformer
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable

from scaledp.schemas.Box import Box
from scaledp.schemas.Image import Image
from scaledp.schemas.DetectorOutput import DetectorOutput


class BaseDetector(Transformer, HasInputCol, HasOutputCol, HasKeepInputData, HasDefaultEnum,
                   DefaultParamsReadable, DefaultParamsWritable, HasScoreThreshold, HasColumnValidator,
                   HasModel):

    scaleFactor = Param(Params._dummy(), "scaleFactor",
                      "Scale Factor.",
                      typeConverter=TypeConverters.toFloat)

    def get_params(self):
        return json.dumps({k.name: v for k, v in self.extractParamMap().items()})

    def transform_udf(self, image, params=None):
        logging.info("Run OCR")
        if params is None:
            params = self.get_params()
        params = json.loads(params)
        if not isinstance(image, Image):
            image = Image(**image.asDict())
        if image.exception != "":
            return DetectorOutput(path=image.path,
                                  text="",
                                  bboxes=[],
                                  type="detector",
                                  exception=image.exception)
        try:
            image_pil = image.to_pil()
            scale_factor = self.getScaleFactor()
            if scale_factor != 1.0:
                resized_image = image_pil.resize((int(image_pil.width * scale_factor), int(image_pil.height * scale_factor)))
            else:
                resized_image = image_pil

            result = self.call_detector([(resized_image, image.path)], params)
        except Exception as e:
            exception = traceback.format_exc()
            exception = f"{self.uid}: Error in object detection: {exception}, {image.exception}"
            logging.warning(f"{self.uid}: Error in object detection.")
            return DetectorOutput(path=image.path,
                                  bboxes=[],
                                  type="detector",
                                  exception=exception)
        return result[0]

    @classmethod
    def call_detector(cls, resized_images, params):
        raise NotImplementedError("Subclasses should implement this method")

    def _transform(self, dataset):
        out_col = self.getOutputCol()
        input_col = self._validate(self.getInputCol(), dataset)
        params = self.get_params()

        result = dataset.withColumn(out_col, udf(self.transform_udf, DetectorOutput.get_schema())(input_col, lit(params)))

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
