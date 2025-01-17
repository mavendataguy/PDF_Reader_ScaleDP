from pyspark import keyword_only
from pyspark.sql.functions import udf, lit
from pyspark.ml import Transformer
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from scaledp.schemas.Image import Image
from scaledp.params import *
from scaledp.enums import ImageType
import logging
import traceback


class ImageError(Exception):
    pass


class DataToImage(
    Transformer,
    HasInputCol,
    HasOutputCol,
    HasKeepInputData,
    HasImageType,
    HasDefaultEnum,
    HasPathCol,
    DefaultParamsReadable,
    DefaultParamsWritable,
    HasColumnValidator,
    HasPropagateError,
):
    """
    Transform Binary Content to Image
    """

    defaultParams = {
        "inputCol": "content",
        "outputCol": "image",
        "pathCol": "path",
        "keepInputData": False,
        "imageType": ImageType.FILE,
        "propagateError": False,
    }

    @keyword_only
    def __init__(self, **kwargs):
        super(DataToImage, self).__init__()
        self._setDefault(**self.defaultParams)
        self._set(**kwargs)

    def transform_udf(self, input, path, resolution):
        try:
            return Image.from_binary(input, path, self.getImageType(), resolution=resolution)
        except Exception as ex:
            exception = traceback.format_exc()
            exception = f"DataToImage: {exception}"
            logging.warning(exception)
            if self.getPropagateError():
                raise ImageError(exception) from ex
            return Image(path, self.getImageType(), data=bytes(), exception=exception)


    def _transform(self, dataset):
        out_col = self.getOutputCol()
        input_col = self._validate(self.getInputCol(), dataset)
        path_col = self._validate(self.getPathCol(), dataset)
        if "resolution" in dataset.columns:
            resolution = dataset["resolution"]
        else:
            resolution = lit(0)
        result = dataset.withColumn(
            out_col,
            udf(self.transform_udf, Image.get_schema())(input_col, path_col, resolution),
        )
        if not self.getKeepInputData():
            result = result.drop(input_col)
        return result
