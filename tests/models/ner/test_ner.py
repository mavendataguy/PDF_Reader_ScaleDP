from sparkpdf.enums import Device
from sparkpdf.models.ner.Ner import Ner
from sparkpdf.models.recognizers.TesseractOcr import TesseractOcr
#from sparkpdf.image.ImageDrawBoxes import ImageDrawBoxes

import pyspark.sql.functions as f
import pytest
import logging
from pyspark.ml.pipeline import PipelineModel

def test_ner(image_df):
    ocr = TesseractOcr()
    ner = Ner(model="obi/deid_bert_i2b2", numPartitions=0, device=Device.CPU.value)
    result_df = ner.transform(ocr.transform(image_df))

    result = result_df.select("ner").cache()
    result_df.select("text.exception").show(1, False)
    data = result.collect()
    assert (len(data) == 1)
    # present ner field
    assert (hasattr(data[0], "ner"))
    ner_tags = result.select(f.explode("ner.entities").alias("ner")).select("ner.*")
    ner_tags.show(40)
    assert (ner_tags.count() > 70)
