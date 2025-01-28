from scaledp.enums import Device
from scaledp.models.recognizers.DocTROcr import DocTROcr


def test_doctr_ocr(image_line_df):
    ocr = DocTROcr(device=Device.CPU, keepFormatting=True)

    # Transform the image dataframe through the OCR stage
    result = ocr.transform(image_line_df).collect()

    # Verify the pipeline result
    assert len(result) == 1, "Expected exactly one result"

    # Verify the detected text contains the expected substring
    assert (
        "24/11/16 08:12:24 WARN Utils:" in result[0].text.text
    ), "Expected text not found in result"


def test_doctr_ocr_pandas_udf(image_line_df):
    ocr = DocTROcr(device=Device.CPU, keepFormatting=True, partitionMap=True)

    # Transform the image dataframe through the OCR stage
    result = ocr.transform(image_line_df).collect()

    # Verify the pipeline result
    assert len(result) == 1, "Expected exactly one result"

    # Verify the detected text contains the expected substring
    assert (
        "24/11/16 08:12:24 WARN Utils:" in result[0].text.text
    ), "Expected text not found in result"
