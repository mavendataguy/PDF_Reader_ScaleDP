
import traceback
import logging
import random

from PIL import ImageDraw
from pyspark import keyword_only
from pyspark.sql.functions import udf
from pyspark.ml import Transformer
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable

from sparkpdf.schemas.Box import Box
from sparkpdf.schemas.Entity import Entity
from sparkpdf.schemas.Image import Image
from sparkpdf.schemas.NerOutput import NerOutput
from ..enums import ImageType
from sparkpdf.params import *


class ImageDrawBoxes(Transformer, HasInputCols, HasOutputCol, HasKeepInputData, HasImageType, HasPageCol,
                     DefaultParamsReadable, DefaultParamsWritable, HasColor, HasNumPartitions, HasColumnValidator):
    """
    Draw boxes on image
    """

    filled = Param(Params._dummy(), "filled",
                      "Fill rectangle.",
                      typeConverter=TypeConverters.toBoolean)

    lineWidth = Param(Params._dummy(), "lineWidth",
                      "Line width.",
                      typeConverter=TypeConverters.toInt)

    textSize = Param(Params._dummy(),"textSize",
                     "Text size.",
                     typeConverter=TypeConverters.toInt)

    padding = Param(Params._dummy(), "padding",
                     "Padding.",
                     typeConverter=TypeConverters.toInt)

    displayDataList = Param(Params._dummy(), "displayDataList",
                      "Display data list.",
                      typeConverter=TypeConverters.toListString)

    @keyword_only
    def __init__(self,
                 inputCols=['image', 'boxes'],
                 outputCol='image_with_boxes',
                 keepInputData=False,
                 imageType=ImageType.FILE.value,
                 filled=False,
                 color=None,
                 lineWidth=1,
                 textSize=12,
                 displayDataList=[],
                 numPartitions=0,
                 padding=0,
                 pageCol="page"):
        super(ImageDrawBoxes, self).__init__()
        self._setDefault(inputCols=inputCols,
                         outputCol=outputCol,
                         keepInputData=keepInputData,
                         imageType=imageType,
                         filled=filled,
                         lineWidth=lineWidth,
                         textSize=textSize,
                         displayDataList=displayDataList,
                         color=color,
                         padding=padding,
                         numPartitions=numPartitions,
                         pageCol=pageCol)
    def getDisplayText(self, box):
        display_list = self.getDisplayDataList()
        text = []
        if display_list:
            for name in display_list:
                if hasattr(box, name):
                    val = getattr(box, name)
                    if isinstance(val, float):
                        text.append(f"{val:0.2f}")
                    elif isinstance(val, int):
                        text.append(str(val))
                    else:
                        text.append(val)
        return ":".join(text)

    def transform_udf(self, image, data):

        def get_color():
            return "#{:06x}".format(random.randint(0, 0xFFFFFF))

        if not isinstance(image, Image):
            image = Image(**image.asDict())
        try:
            if image.exception != "":
                return Image(image.origin, image.imageType, data=bytes(), exception=image.exception)
            img = image.to_pil()
            img1 = ImageDraw.Draw(img)
            fill = self.getColor() if self.getFilled()  else None

            if hasattr(data, "entities"):
                if not isinstance(data, NerOutput):
                    data = NerOutput(**data.asDict())
                colors = {}
                for ner in data.entities:

                    if not isinstance(ner, Entity):
                        ner = Entity(**ner.asDict())

                    if ner.entity_group not in colors:
                        colors[ner.entity_group] = get_color()
                    if self.getColor() is None:
                        color = colors[ner.entity_group]
                    else:
                        color = self.getColor()
                    for box in ner.boxes:
                        if not isinstance(box, Box):
                            box = Box(**box.asDict())
                        text = self.getDisplayText(ner)
                        img1.rounded_rectangle(box.shape(self.getPadding()), outline=color, radius=4, fill=fill, width=self.getLineWidth())
                        if text:
                            tbox = list(img1.textbbox((box.x, box.y - self.getTextSize() * 1.2 - self.getPadding()), text , font_size=self.getTextSize()))
                            tbox[3] = tbox[3] + self.getTextSize()/4
                            tbox[2] = tbox[2] + self.getTextSize() / 4
                            tbox[0] = box.x - self.getPadding()
                            tbox[1] = box.y - self.getTextSize() * 1.2 - self.getPadding()
                            img1.rounded_rectangle(tbox, outline=color, radius=2, fill=color)
                            img1.text((box.x, box.y - self.getTextSize() * 1.2 - self.getPadding()), text,
                                      stroke_width=0, fill="white", font_size=self.getTextSize())
            else:
                for box in data.bboxes:
                    if not isinstance(box, Box):
                        box = Box(**box.asDict())
                    img1.rounded_rectangle(box.shape(self.getPadding()), outline=self.getColor(), radius=4, fill=fill, width=self.getLineWidth())
                    text = self.getDisplayText(box)
                    if text:
                        img1.text((box.x, box.y - self.getTextSize() * 1.2 - self.getPadding()), text, fill=self.getColor(), font_size=self.getTextSize())


        except Exception as e:
            exception = traceback.format_exc()
            exception = f"ImageDrawBoxes: {exception}, {image.exception}"
            logging.warning(exception)
            return Image(image.path, image.imageType, data=bytes(), exception=exception)
        return Image.from_pil(img, image.path, image.imageType, image.resolution)

    def _transform(self, dataset):
        out_col = self.getOutputCol()
        image_col = self._validate(self.getInputCols()[0], dataset)
        box_col = self._validate(self.getInputCols()[1], dataset)

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

        meta private:
        """
        return self.getOrDefault(self.filled)

    def setLineWidth(self, value):
        """
        Sets the value of :py:attr:`lineWidth`.
        """
        return self._set(lineWidth=value)

    def getLineWidth(self):
        """
        Gets the value of :py:attr:`lineWidth`.
        """
        return self.getOrDefault(self.lineWidth)

    def setTextSize(self, value):
        """
        Sets the value of :py:attr:`textSize`.
        """
        return self._set(textSize=value)

    def getTextSize(self):
        """
        Gets the value of :py:attr:`textSize`.
        """
        return self.getOrDefault(self.textSize)

    def setDisplayDataList(self, value):
        """
        Sets the value of :py:attr:`displayDataList`.
        """
        return self._set(displayDataList=value)

    def getDisplayDataList(self):
        """
        Gets the value of :py:attr:`displayDataList`.
        """
        return self.getOrDefault(self.displayDataList)

    def getPadding(self):
        """
        Gets the value of :py:attr:`padding`.
        """
        return self.getOrDefault(self.padding)
