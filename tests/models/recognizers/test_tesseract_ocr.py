import pytest
from pyspark import Row

from sparkpdf.enums import TessLib, PSM
from sparkpdf.image.DataToImage import DataToImage
from sparkpdf.models.recognizers.TesseractOcr import TesseractOcr

def test_tesseract_ocr(image_df):
    pytest.skip()
    # Initialize the Tesseract OCR stage with specific parameters
    ocr = TesseractOcr(keepFormatting=False, tessLib=TessLib.TESSEROCR.value)

    # Transform the image dataframe through the OCR stage
    result = ocr.transform(image_df).collect()

    # Verify the pipeline result
    assert len(result) == 1, "Expected exactly one result"

    # Verify the presence of the 'text' field in the result
    assert hasattr(result[0], "text"), "Expected 'text' field in the result"

    # Verify the detected text contains the expected substring
    assert "Hospital:" in result[0].text.text, "Expected 'Hospital:' in the detected text"

def test_tesseract_ocr_pytesseract(image_df):
    # Initialize the Tesseract OCR stage with specific parameters
    ocr = TesseractOcr(keepFormatting=True, psm=PSM.AUTO)

    # Transform the image dataframe through the OCR stage
    result = ocr.transform(image_df).collect()

    # Verify the pipeline result
    assert len(result) == 1, "Expected exactly one result"

    # Verify the presence of the 'text' field in the result
    assert hasattr(result[0], "text"), "Expected 'text' field in the result"

    # Verify the detected text contains the expected substring
    assert "Hospital:" in result[0].text.text, "Expected 'Hospital:' in the detected text"

def test_wrong_file_tesseract_ocr(pdf_df):
    # Initialize the DataToImage stage to convert PDF to image
    data_to_image = DataToImage()

    # Initialize the Tesseract OCR stage
    ocr = TesseractOcr()

    # Transform the PDF dataframe to image dataframe
    image_df = data_to_image.transform(pdf_df)

    # Transform the image dataframe through the OCR stage
    result = ocr.transform(image_df).collect()

    # Verify the pipeline result
    assert len(result) == 1, "Expected exactly one result"

    # Verify the presence of the 'text' field in the result
    assert hasattr(result[0], "text"), "Expected 'text' field in the result"

    # Verify that the exception message is as expected
    assert "Unable to read image" in result[
        0].text.exception, "Expected 'Unable to read image' in the exception message"

def test_tesseract_ocr_class(image):
    # Test TesseractOcr with keepFormatting=False
    result1 = TesseractOcr(keepFormatting=False).transform_udf(image)
    assert result1.exception == "", "Expected no exception for keepFormatting=False"

    # Test TesseractOcr with keepFormatting=True and scaleFactor=1.2
    result2 = TesseractOcr(keepFormatting=True, scaleFactor=1.2).transform_udf(image)
    assert result2.exception == "", "Expected no exception for keepFormatting=True and scaleFactor=1.2"

    # Test TesseractOcr with keepFormatting=True and tessLib=TESSEROCR
    # TODO: temporary commented while fixing the issue on github tests
    # result3 = TesseractOcr(keepFormatting=True, tessLib=TessLib.TESSEROCR.value).transform_udf(image)
    # assert result3.exception == "", "Expected no exception for keepFormatting=True and tessLib=TESSEROCR"

    # Test TesseractOcr with keepFormatting=True and tessLib=2
    result4 = TesseractOcr(keepFormatting=True, tessLib=2).transform_udf(image)
    assert result4.exception != "", "Expected an exception for keepFormatting=True and tessLib=2"

    # Test TesseractOcr with an image that has an exception
    image_with_exception = Row(path="test", exception="test exception")
    result5 = TesseractOcr(keepFormatting=True).transform_udf(image_with_exception)
    assert "test exception" in result5.exception, "Expected 'test exception' in the result exception"

