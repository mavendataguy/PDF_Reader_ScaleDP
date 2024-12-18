from scaledp.models.recognizers.BaseOcr import BaseOcr
from scaledp.params import HasInputCols
from pyspark.sql.functions import udf, pandas_udf, lit

from scaledp.schemas.DetectorOutput import DetectorOutput
from scaledp.schemas.Document import Document
from scaledp.schemas.Image import Image
import logging
import json
import traceback


class BaseRecognizer(BaseOcr, HasInputCols):

    def transform_udf(self, image, boxes, params=None):
        logging.info("Run Text Recognizer")
        if params is None:
            params = self.get_params()
        params = json.loads(params)
        if not isinstance(image, Image):
            image = Image(**image.asDict())

        if not isinstance(boxes, DetectorOutput):
            boxes = DetectorOutput(**boxes.asDict())
        if image.exception != "":
            return Document(path=image.path,
                            text="",
                            bboxes=[],
                            type="text",
                            exception=image.exception)
        try:
            image_pil = image.to_pil()
            scale_factor = self.getScaleFactor()
            if scale_factor != 1.0:
                resized_image = image_pil.resize((int(image_pil.width * scale_factor), int(image_pil.height * scale_factor)))
            else:
                resized_image = image_pil

            result = self.call_recognizer([(resized_image, image.path)], [boxes], params)
        except Exception as e:
            exception = traceback.format_exc()
            exception = f"{self.uid}: Error in text recognition: {exception}, {image.exception}"
            logging.warning(f"{self.uid}: Error in text recognition.")
            return Document(path=image.path,
                            text="",
                            bboxes=[],
                            type="ocr",
                            exception=exception)
        return result[0]

    @classmethod
    def call_recognizer(cls, resized_images, boxes, params):
        raise NotImplementedError("Subclasses should implement this method")

    def _transform(self, dataset):
        out_col = self.getOutputCol()
        image_col = self._validate(self.getInputCols()[0], dataset)
        box_col = self._validate(self.getInputCols()[1], dataset)
        params = self.get_params()

        result = dataset.withColumn(out_col, udf(self.transform_udf, Document.get_schema())(image_col, box_col, lit(params)))

        if not self.getKeepInputData():
            result = result.drop(image_col)
        return result
