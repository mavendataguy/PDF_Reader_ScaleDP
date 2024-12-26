from pyspark.sql.functions import input_file_name, udf
from pyspark.ml import Transformer
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from scaledp.params import *
from scaledp.schemas.Document import Document


class TextToDocument(
    Transformer, HasInputCol, HasOutputCol, DefaultParamsReadable, DefaultParamsWritable
):

    def __init__(self, inputCol="value", outputCol="text"):
        super(TextToDocument, self).__init__()
        self._setDefault(inputCol=inputCol, outputCol=outputCol)

    @staticmethod
    def transform_udf(text, path):
        return Document(
            path=path,
            text=text,
            type="text",
            bboxes=[],
        )

    def _transform(self, dataset):
        input_col = self.getInputCol()
        output_col = self.getOutputCol()

        text_to_document_udf = udf(self.transform_udf, Document.get_schema())
        return dataset.withColumn(
            output_col, text_to_document_udf(dataset[input_col], input_file_name())
        )
