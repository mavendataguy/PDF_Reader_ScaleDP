
from pyspark.ml.pipeline import PipelineModel
import pytest
import tempfile

from sparkpdf.image.ImageDrawBoxes import ImageDrawBoxes
from sparkpdf.models.recognizers.TesseractOcr import TesseractOcr
from sparkpdf.models.ner.Ner import Ner


def test_image_draw_boxes_ocr(image_df):

    pipeline = PipelineModel(stages=[
        TesseractOcr(keepInputData=True),
        ImageDrawBoxes(inputCols=["image", "text"], lineWidth=2, textSize=20)
    ])

    result = pipeline.transform(image_df).collect()
    assert (len(result) == 1)
    # present image_with_boxes field
    assert (hasattr(result[0], "image_with_boxes"))

    temp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    print("file://" + temp.name)
    temp.write(result[0].image_with_boxes.data)
    temp.close()


def test_image_draw_boxes_ner(image_df):

    pipeline = PipelineModel(stages=[
        TesseractOcr(keepInputData=True),
        Ner(outputCol="ner", model="obi/deid_bert_i2b2"),
        ImageDrawBoxes(inputCols=["image", "ner"], filled=False, color="red")
    ])

    result = pipeline.transform(image_df).collect()
    assert (len(result) == 1)
    # present image_with_boxes
    assert (hasattr(result[0], "image_with_boxes"))

    temp = tempfile.NamedTemporaryFile(suffix=".webp", delete=False)
    print("file://" + temp.name)
    temp.write(result[0].image_with_boxes.data)
    temp.close()
