import pytest
from pyspark import Row

from sparkpdf.enums import TessLib
from sparkpdf.image.DataToImage import DataToImage
from sparkpdf.models.recognizers.TesseractOcr import TesseractOcr

def test_tesseract_ocr(image_df):
    pytest.skip()
    ocr = TesseractOcr(keepFormatting=False, tessLib=TessLib.TESSEROCR.value)
    result = ocr.transform(image_df).collect()
    assert (len(result) == 1)
    # present text field
    assert (hasattr(result[0], "text"))
    # detected text
    assert ("Hospital:" in result[0].text.text)

def test_tesseract_ocr_pytesseract(image_df):
    ocr = TesseractOcr(keepFormatting=True)
    result = ocr.transform(image_df).collect()
    assert (len(result) == 1)
    # present text field
    assert (hasattr(result[0], "text"))
    # detected text
    assert ("Hospital:" in result[0].text.text)

def test_wrong_file_tesseract_ocr(pdf_df):
    data_to_image = DataToImage()
    ocr = TesseractOcr()
    result = ocr.transform(data_to_image.transform(pdf_df)).collect()
    assert (len(result) == 1)
    # present text field
    assert (hasattr(result[0], "text"))
    assert ("Unable to read image" in result[0].text.exception)

def test_tesseract_ocr_class(image):
    TesseractOcr(keepFormatting=False).transform_udf(image)
    TesseractOcr(keepFormatting=True, scaleFactor=1.2).transform_udf(image)
    #TesseractOcr(keepFormatting=True, tessLib=TessLib.TESSEROCR.value).transform_udf(image)
    assert( TesseractOcr(keepFormatting=True, tessLib=2).transform_udf(image).exception != "" )
    image.exception = "test exception"
    assert("test exception" in TesseractOcr(keepFormatting=True). \
        transform_udf(Row(path="test", exception="test exception")).exception )
