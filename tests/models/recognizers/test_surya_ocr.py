import pytest
from sparkpdf.models.recognizers.SuryaOcr import SuryaOcr
from sparkpdf.enums import Device


def test_surya_ocr_class(image, image_pil):
    pytest.skip("Slow test")
    # Test SuryatOcr
    result1 = SuryaOcr(keepFormatting=False, partitionMap=False, device=Device.CUDA).transform_udf(image)
    assert result1.exception == "", "Expected no exception for keepFormatting=False"
    assert "Hospital:" in result1.text, "Expected text for keepFormatting=False"


def test_surya_ocr_pandas_udf(image_df):
    pytest.skip("Slow test")
    ocr = SuryaOcr(partitionMap=True, device=Device.CUDA, keepFormatting=True)

    # Transform the image dataframe through the OCR stage
    result = ocr.transform(image_df).collect()

    # Verify the pipeline result
    assert len(result) == 1, "Expected exactly one result"

    # Verify the detected text contains the expected substring
    assert "Hospital:" in result[0].text.text, "Expected 'Hospital:' in the detected text"
