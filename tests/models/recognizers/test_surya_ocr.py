import pytest
from pyspark.sql import DataFrame

from scaledp.enums import Device
from scaledp.models.recognizers.SuryaOcr import SuryaOcr


def test_surya_ocr_class(image_line: DataFrame) -> None:
    pytest.skip("Slow test")
    # Test SuryatOcr
    result1 = SuryaOcr(
        keepFormatting=False,
        partitionMap=False,
        device=Device.CPU,
    ).transform_udf(image_line)
    assert result1.exception == "", "Expected no exception for keepFormatting=False"
    assert "24/11/16 08:12:24 WARN Utils: Service 'SparkUI'." in result1.text


def test_surya_ocr_pandas_udf(image_line_df: DataFrame) -> None:
    pytest.skip("Slow test")
    ocr = SuryaOcr(partitionMap=True, device=Device.CPU, keepFormatting=True)

    # Transform the image dataframe through the OCR stage
    result = ocr.transform(image_line_df).collect()

    # Verify the pipeline result
    assert len(result) == 1, "Expected exactly one result"

    # Verify the detected text contains the expected substring
    assert "24/11/16 08:12:24 WARN Utils: Service 'SparkUI'" in result[0].text.text
