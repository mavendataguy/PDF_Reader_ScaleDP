import pytest
from scaledp.models.recognizers.LLMOcr import LLMOcr


def test_llm_ocr_class(image_line):
    # Test SuryatOcr
    result1 = LLMOcr(keepFormatting=False, partitionMap=False).transform_udf(image_line)
    assert result1.exception == "", "Expected no exception for keepFormatting=False"
    assert "24/11/16 08:12:24 WARN Utils: Service 'SparkUI' could not bind on port 4043. Attempting port 4044." in result1.text
