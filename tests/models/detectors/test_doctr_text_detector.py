import pytest
import tempfile
from scaledp import DataToImage, ImageDrawBoxes
from scaledp.models.detectors.DocTRTextDetector import DocTRTextDetector
from scaledp.enums import Device


def test_doctr_text_detector(image_receipt_df):
    detector = DocTRTextDetector(device=Device.CPU, keepInputData=True, scoreThreshold=0.2)

    draw = ImageDrawBoxes(keepInputData=True, inputCols=["image", "boxes"],
                          filled=False, color="green", lineWidth=2,
                          displayDataList=['score'])
    # Transform the image dataframe through the OCR stage
    result = draw.transform(detector.transform(image_receipt_df)).cache()

    data = result.collect()

    # Verify the pipeline result
    assert len(data) == 1, "Expected exactly one result"

    # Check that exceptions is empty
    assert data[0].boxes.exception == ""

    # Save the output image to a temporary file for verification
    temp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    temp.write(data[0].image_with_boxes.data)
    temp.close()

    # Print the path to the temporary file
    print("file://" + temp.name)