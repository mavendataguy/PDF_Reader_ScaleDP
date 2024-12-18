import pytest
import tempfile
from scaledp import DataToImage, ImageDrawBoxes
from scaledp.models.detectors.DocTRTextDetector import DocTRTextDetector
from scaledp.enums import Device, TessLib
from scaledp.models.recognizers.TesseractRecognizer import TesseractRecognizer


def test_tesseract_recognizer(image_receipt_df):
    detector = DocTRTextDetector(device=Device.CPU, keepInputData=True, scoreThreshold=0.2)

    ocr = TesseractRecognizer(keepFormatting=True,
                              tessLib=TessLib.TESSEROCR.value,
                              lang=["ukr", "eng"],
                              tessDataPath="/usr/local/Cellar/tesseract-lang/4.1.0/share/tessdata/")
    # Transform the image dataframe through the OCR stage
    result = ocr.transform(detector.transform(image_receipt_df)).cache()

    data = result.collect()

    # Verify the pipeline result
    assert len(data) == 1, "Expected exactly one result"

    # Check that exceptions is empty
    assert data[0].text.exception == ""

    print(data[0].text.text)

    assert "Hospital:" in data[0].text.text