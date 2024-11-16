import pytest
from sparkpdf.models.recognizers.EasyOcr import EasyOcr
from sparkpdf.enums import Device


def test_easy_ocr(image_line_df):
    pytest.skip("Slow test")
    ocr = EasyOcr(device=Device.CPU, keepFormatting=True)

    # Transform the image dataframe through the OCR stage
    result = ocr.transform(image_line_df).collect()

    # Verify the pipeline result
    assert len(result) == 1, "Expected exactly one result"

    # Verify the detected text contains the expected substring
    assert ("24/11/16 08:12:24 WARN Utils: Service SparkUI could not bind on port 4043"
            in result[0].text.text), "Expected text not found in result"


def test_easy_ocr_pandas_udf(image_line_df):
    pytest.skip("Slow test")
    ocr = EasyOcr(device=Device.CPU, keepFormatting=True, partitionMap=True)

    # Transform the image dataframe through the OCR stage
    result = ocr.transform(image_line_df).collect()

    # Verify the pipeline result
    assert len(result) == 1, "Expected exactly one result"

    # Verify the detected text contains the expected substring
    assert ("24/11/16 08:12:24 WARN Utils: Service SparkUI could not bind on port 4043"
            in result[0].text.text), "Expected text not found in result"