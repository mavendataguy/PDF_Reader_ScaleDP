

import json
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.sql.types import *
from pyspark.ml import Transformer
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql.functions import pandas_udf, lit, udf

from sparkpdf.params import *
from sparkpdf.schemas.NerOutput import NerOutput


class BaseNer(Transformer, HasInputCol, HasOutputCol, HasKeepInputData, HasWhiteList, HasDevice, HasModel, HasPathCol,
          DefaultParamsReadable, DefaultParamsWritable, HasNumPartitions, HasScoreThreshold, HasBatchSize, HasPageCol):

    def outputSchema(self):
        return StructType([StructField("path", StringType(), True),
                               StructField("entities",
                                           ArrayType(StructType([StructField("entity_group", StringType(), False),
                                                                 StructField("score", DoubleType(), False),
                                                                 StructField("word", StringType(), False),
                                                                 StructField("start", IntegerType(), False),
                                                                 StructField("end", IntegerType(), False),
                                                                 StructField("boxes", ArrayType(StructType(
                                                                     [StructField("text", StringType(), False)
                                                                         , StructField("score", DoubleType(), False),
                                                                      StructField("x", IntegerType(), False),
                                                                      StructField("y", IntegerType(), False),
                                                                      StructField("width", IntegerType(), False),
                                                                      StructField("height", IntegerType(), False)]),
                                                                                                True),
                                                                             False)]),
                                                     True),
                                           True),
                               StructField("exception", StringType(), True)])

    def get_params(self):
        return json.dumps({k.name: v for k, v in self.extractParamMap().items()})

    def _transform(self, dataset):
        params = self.get_params()
        out_col = self.getOutputCol()
        if self.getInputCol() not in dataset.columns:
            input_col = self.getInputCol()
            raise ValueError(f"""Missing input column in {self.uid}: Column '{input_col}' is not present.""")
        in_col = dataset[self.getInputCol()]

        if not hasattr(dataset, "sparkSession"):
            result = dataset.withColumn(out_col,
                                        udf(self.transform_udf, NerOutput.get_schema())
                                        (in_col))
        else:
            if self.getNumPartitions() > 0:
                if self.getPageCol() in dataset.columns:
                    dataset = dataset.repartition( self.getPageCol())
                else:
                    dataset = dataset.repartition(self.getPathCol())
                dataset = dataset.coalesce(self.getNumPartitions())
            result = dataset.withColumn(out_col,
                                        pandas_udf(self.transform_udf_pandas, self.outputSchema())
                                        (in_col, lit(params)))

        if not self.getKeepInputData():
            result = result.drop(in_col)
        return result


