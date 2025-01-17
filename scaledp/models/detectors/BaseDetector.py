import json
import traceback
import logging

import pandas as pd
from pyspark.sql.functions import udf, pandas_udf, lit
from pyspark.sql.types import ArrayType

from scaledp.params import *
from pyspark.sql.types import *
from pyspark.ml import Transformer
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable

from scaledp.schemas.Image import Image
from scaledp.schemas.DetectorOutput import DetectorOutput


class DetectionError(Exception):
    pass

class BaseDetector(
    Transformer,
    HasInputCol,
    HasOutputCol,
    HasKeepInputData,
    HasDefaultEnum,
    DefaultParamsReadable,
    DefaultParamsWritable,
    HasScoreThreshold,
    HasColumnValidator,
    HasModel,
    HasPartitionMap,
    HasNumPartitions,
    HasPageCol,
    HasPathCol,
    HasPropagateError
):

    scaleFactor = Param(
        Params._dummy(),
        "scaleFactor",
        "Scale Factor.",
        typeConverter=TypeConverters.toFloat,
    )

    def get_params(self):
        return json.dumps({k.name: v for k, v in self.extractParamMap().items()})

    def outputSchema(self):
        return StructType(
            [
                StructField("path", StringType(), True),
                StructField("type", StringType(), True),
                StructField(
                    "bboxes",
                    ArrayType(
                        StructType(
                            [
                                StructField("text", StringType(), False),
                                StructField("score", DoubleType(), False),
                                StructField("x", IntegerType(), False),
                                StructField("y", IntegerType(), False),
                                StructField("width", IntegerType(), False),
                                StructField("height", IntegerType(), False),
                            ]
                        ),
                        True,
                    ),
                    True,
                ),
                StructField("exception", StringType(), True),
            ]
        )

    def transform_udf(self, image, params=None):
        logging.info("Run Detector")
        if params is None:
            params = self.get_params()
        params = json.loads(params)
        if not isinstance(image, Image):
            image = Image(**image.asDict())
        if image.exception != "":
            return DetectorOutput(
                path=image.path,
                bboxes=[],
                type="detector",
                exception=image.exception,
            )
        try:
            image_pil = image.to_pil()
            scale_factor = self.getScaleFactor()
            if scale_factor != 1.0:
                resized_image = image_pil.resize(
                    (
                        int(image_pil.width * scale_factor),
                        int(image_pil.height * scale_factor),
                    )
                )
            else:
                resized_image = image_pil

            result = self.call_detector([(resized_image, image.path)], params)
        except Exception as e:
            exception = traceback.format_exc()
            exception = f"{self.uid}: Error in object detection: {exception}, {image.exception}"
            logging.warning(f"{self.uid}: Error in object detection.")
            if self.getPropagateError():
                raise DetectionError() from e
            return DetectorOutput(path=image.path, bboxes=[], type="detector", exception=exception)
        return result[0]

    @classmethod
    def call_detector(cls, resized_images, params):
        raise NotImplementedError("Subclasses should implement this method")

    @classmethod
    def transform_udf_pandas(cls, images: pd.DataFrame, params: pd.Series) -> pd.DataFrame:
        params = json.loads(params[0])
        resized_images = []
        for index, image in images.iterrows():
            if not isinstance(image, Image):
                image = Image(**image.to_dict())
            image_pil = image.to_pil()
            scale_factor = params["scaleFactor"]
            if scale_factor != 1.0:
                resized_image = image_pil.resize(
                    (
                        int(image_pil.width * scale_factor),
                        int(image_pil.height * scale_factor),
                    )
                )
            else:
                resized_image = image_pil
            resized_images.append((resized_image, image.path))

        results = cls.call_detector(resized_images, params)

        return pd.DataFrame(results)

    def _transform(self, dataset):
        out_col = self.getOutputCol()
        input_col = self._validate(self.getInputCol(), dataset)
        params = self.get_params()

        if not self.getPartitionMap():
            result = dataset.withColumn(
                out_col,
                udf(self.transform_udf, DetectorOutput.get_schema())(input_col, lit(params)),
            )
        else:
            if self.getNumPartitions() > 0:
                if self.getPageCol() in dataset.columns:
                    dataset = dataset.repartition(self.getPageCol())
                elif self.getPathCol() in dataset.columns:
                    dataset = dataset.repartition(self.getPathCol())
                dataset = dataset.coalesce(self.getNumPartitions())
            result = dataset.withColumn(
                out_col,
                pandas_udf(self.transform_udf_pandas, self.outputSchema())(input_col, lit(params)),
            )

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
