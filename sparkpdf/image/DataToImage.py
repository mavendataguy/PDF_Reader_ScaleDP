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

    def transform_udf(self, input, origin, resolution):
        return Image.from_binary(input, origin, self.getImageType(), resolution=resolution)

    def _transform(self, dataset):
        out_col = self.getOutputCol()
        if self.getInputCol() not in dataset.columns:
            uid_ = self.uid
            inpcol = self.getInputCol()
            raise ValueError(f"""Missing input column in {uid_}: Column '{inpcol}' is not present.
                                 Make sure such transformer exist in your pipeline,
                                 with the right output names.""")
        in_col = dataset[self.getInputCol()]
        origin_col = dataset[self.getPathCol()]
        if "resolution" in dataset.columns:
            resolution = dataset["resolution"]
        else:
            resolution = lit(0)
        result = dataset.withColumn(out_col, udf(self.transform_udf, Image.get_schema())(in_col, origin_col, resolution))
        if not self.getKeepInputData():
            result = result.drop(in_col)
        return result