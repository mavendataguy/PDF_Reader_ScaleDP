
import traceback
import logging
from PIL import ImageDraw
from pyspark import keyword_only
from pyspark.sql.functions import udf
from pyspark.ml import Transformer
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable

from sparkpdf.schemas.Box import Box
from sparkpdf.schemas.Entity import Entity
from sparkpdf.schemas.Image import Image
from sparkpdf.schemas.NerOutput import NerOutput
from enums import ImageType
from sparkpdf.params import *


class ImageDrawBoxes(Transformer, HasInputCols, HasOutputCol, HasKeepInputData, HasImageType, HasPageCol,
                     DefaultParamsReadable, DefaultParamsWritable, HasColor, HasNumPartitions):
    """
    Draw boxes on image
    """

    filled = Param(Params._dummy(), "filled",
                      "Fill rectangle.",
                      typeConverter=TypeConverters.toBoolean)

    @keyword_only
    def __init__(self,
                 inputCols=['image', 'boxes'],
                 outputCol='image_with_boxes',
                 keepInputData=False,
                 imageType=ImageType.FILE.value,
                 filled=False,
                 color="red",
                 numPartitions=0,
                 pageCol="page"):
        super(ImageDrawBoxes, self).__init__()
        self._setDefault(inputCols=inputCols,
                         outputCol=outputCol,
                         keepInputData=keepInputData,
                         imageType=imageType,
                         filled=filled,
                         color=color,
                         numPartitions=numPartitions,
                         pageCol=pageCol)

    def transform_udf(self, image, data):
        if not isinstance(image, Image):
            image = Image(**image.asDict())
        try:
            if image.exception != "":
                return Image(image.origin, image.imageType, data=bytes(), exception=image.exception)
            img = image.to_pil()
            img1 = ImageDraw.Draw(img)
            fill = self.getColor() if self.getFilled()  else None
            if "entities" in data:
                if not isinstance(data, NerOutput):
                    data = NerOutput(**data.asDict())
                for ner in data.entities:
                    if not isinstance(ner, Entity):
                        ner = Entity(**ner.asDict())
                    for box in ner.boxes:
                        if not isinstance(box, Box):
                            box = Box(**box.asDict())
                        img1.rectangle(box.shape(), outline=self.getColor(), fill=fill)
                        img1.text((box.x, box.y - box.height), ner.entity_group, fill=self.getColor())
            else:
                for box in data.bboxes:
                    box = Box(**box.asDict())
                    img1.rectangle(box.shape(), outline=self.getColor(), fill=fill)
                    img1.text((box.x, box.y - 14), str(f"{box.score:02f}"), fill=self.getColor(), font_size=12)

        except Exception as e:
            exception = traceback.format_exc()
            exception = f"ImageDrawBoxes: {exception}, {image.exception}"
            logging.warning(exception)
            return Image(image.path, image.imageType, data=bytes(), exception=exception)
        return Image.from_pil(img, image.path, image.imageType, image.resolution)

    def _transform(self, dataset):
        out_col = self.getOutputCol()
        if self.getInputCols()[0] not in dataset.columns:
            input_col = self.getInputCols()[0]
            raise ValueError(f"""Missing input column in {self.uid}: Column '{input_col}' is not present.""")
        image_col = dataset[self.getInputCols()[0]]
        box_col = dataset[self.getInputCols()[1]]

        if self.getNumPartitions() > 0:
            dataset = dataset.repartition(self.getPageCol()).coalesce(self.getNumPartitions())
        result = dataset.withColumn(out_col, udf(self.transform_udf, Image.get_schema())(image_col, box_col))

        if not self.getKeepInputData():
            result = result.drop(image_col)
        return result

    def setFilled(self, value):
        """
        Sets the value of :py:attr:`filled`.
        """
        return self._set(filled=value)

    def getFilled(self):
        """
        Gets the value of :py:attr:`filled`.

        :meta private:
        """
        return self.getOrDefault(self.filled)
