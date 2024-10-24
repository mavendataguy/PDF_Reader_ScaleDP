from pyspark import keyword_only
from pyspark.sql.functions import udf, lit
from pyspark.ml import Transformer
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from sparkpdf.schemas.Image import Image
from sparkpdf.params import *
from sparkpdf.enums import ImageType


class DataToImage(Transformer, HasInputCol, HasOutputCol, HasKeepInputData, HasImageType,
                  HasPathCol, DefaultParamsReadable, DefaultParamsWritable):
    """
    Transform Binary Content to Image
    """

    @keyword_only
    def __init__(self):
        super(DataToImage, self).__init__()
        self._setDefault(outputCol='image')
        self._setDefault(inputCol='content')
        self._setDefault(pathCol="path")
        self._setDefault(keepInputData=False)
        self._setDefault(imageType=ImageType.FILE.value)

    def transform_udf(self, input, path, resolution):
        return Image.from_binary(input, path, self.getImageType(), resolution=resolution)

    def _transform(self, dataset):
        out_col = self.getOutputCol()
        if self.getInputCol() not in dataset.columns:
            input_col = self.getInputCol()
            raise ValueError(f"Missing input column in transformer {self.uid}: Column '{input_col}' is not present.")
        input_col = dataset[self.getInputCol()]
        path_col = dataset[self.getPathCol()]
        if "resolution" in dataset.columns:
            resolution = dataset["resolution"]
        else:
            resolution = lit(0)
        result = dataset.withColumn(out_col, udf(self.transform_udf, Image.get_schema())(input_col, path_col, resolution))
        if not self.getKeepInputData():
            result = result.drop(input_col)
        return result
