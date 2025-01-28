import tempfile

from pyspark.ml import PipelineModel

from scaledp import ImageDrawBoxes
from scaledp.enums import Device
from scaledp.models.detectors.DocTRTextDetector import DocTRTextDetector


def test_doctr_text_detector(image_receipt_df):
    pipeline = create_pipeline(partitionMap=False)

    result = pipeline.transform(image_receipt_df).cache()
    data = result.collect()

    # Verify the pipeline result
    assert len(data) == 1, "Expected exactly one result"

    # Check that exceptions is empty
    assert data[0].boxes.exception == ""

    # Save the output image to a temporary file for verification
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp:
        temp.write(data[0].image_with_boxes.data)
        temp.close()

        # Print the path to the temporary file
        print("file://" + temp.name)


def test_doctr_text_detector_pandas(image_receipt_df):
    pipeline = create_pipeline()
    # Transform the image dataframe through the OCR stage
    result = pipeline.transform(image_receipt_df).cache()

    data = result.collect()

    # Verify the pipeline result
    assert len(data) == 1, "Expected exactly one result"

    # Check that exceptions is empty
    assert data[0].boxes.exception == ""


def create_pipeline(partitionMap=True):
    detector = DocTRTextDetector(
        device=Device.CPU,
        keepInputData=True,
        scoreThreshold=0.2,
        partitionMap=partitionMap,
    )
    draw = ImageDrawBoxes(
        keepInputData=True,
        inputCols=["image", "boxes"],
        filled=False,
        color="green",
        lineWidth=2,
        displayDataList=["score"],
    )
    return PipelineModel(stages=[detector, draw])
