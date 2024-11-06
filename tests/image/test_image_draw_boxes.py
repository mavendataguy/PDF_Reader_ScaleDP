
from pyspark.ml.pipeline import PipelineModel
import tempfile

from sparkpdf import DataToImage
from sparkpdf.enums import PSM
from sparkpdf.image.ImageDrawBoxes import ImageDrawBoxes
from sparkpdf.models.recognizers.TesseractOcr import TesseractOcr
from sparkpdf.models.ner.Ner import Ner


def test_image_draw_boxes_ocr(image_df):

    # Initialize the OCR stage with specific parameters
    ocr = TesseractOcr(keepInputData=True, scoreThreshold=0.5, psm=PSM.SPARSE_TEXT.value, scaleFactor=2.0)

    # Initialize the ImageDrawBoxes stage with specific parameters
    draw = ImageDrawBoxes(inputCols=["image", "text"], lineWidth=2, textSize=20, displayDataList=["text", "score"])

    # Create the pipeline with the OCR and ImageDrawBoxes stages
    pipeline = PipelineModel(stages=[ocr, draw])

    # Run the pipeline on the input image dataframe
    result = pipeline.transform(image_df).collect()

    # Verify the pipeline result
    assert len(result) == 1
    assert hasattr(result[0], "image_with_boxes")

    # Save the output image to a temporary file for verification
    temp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    temp.write(result[0].image_with_boxes.data)
    temp.close()

    # Print the path to the temporary file
    print("file://" + temp.name)

    # Verify the OCR stage output
    ocr_result = result[0].text
    assert len(ocr_result) > 0

    # Verify the draw stage output
    draw_result = result[0].image_with_boxes
    assert draw_result.exception == ""


def test_image_draw_boxes_ner(image_df):
    # Initialize the pipeline stages
    ocr = TesseractOcr(keepInputData=True)
    ner = Ner(outputCol="ner", model="obi/deid_bert_i2b2")
    draw = ImageDrawBoxes(inputCols=["image", "ner"], filled=False, color="red")

    # Create the pipeline
    pipeline = PipelineModel(stages=[ocr, ner, draw])

    # Run the pipeline on the input image dataframe
    result = pipeline.transform(image_df).collect()

    # Verify the pipeline result
    assert len(result) == 1
    assert hasattr(result[0], "image_with_boxes")

    # Save the output image to a temporary file for verification
    temp = tempfile.NamedTemporaryFile(suffix=".webp", delete=False)
    temp.write(result[0].image_with_boxes.data)
    temp.close()

    # Print the path to the temporary file
    print("file://" + temp.name)

def test_image_draw_boxes_local(image_file, pdf_file):
    from sparkpdf.pipeline.PandasPipeline import LocalPipeline, UserDefinedFunction
    import pyspark

    # Temporarily replace the UserDefinedFunction
    temp = pyspark.sql.udf.UserDefinedFunction
    pyspark.sql.udf.UserDefinedFunction = UserDefinedFunction

    # Initialize the pipeline stages
    data_to_image = DataToImage()
    ocr = TesseractOcr(keepInputData=True)
    draw = ImageDrawBoxes(keepInputData=True, inputCols=["image", "text"],
                          filled=True, color="orange",
                          displayDataList=["text", "score"])

    # Create the pipeline
    pipeline = LocalPipeline(stages=[data_to_image, ocr, draw])

    # Run the pipeline on the input image file
    result = pipeline.fromFile(image_file)

    # Verify the pipeline result
    assert result is not None
    assert "image_with_boxes" in result.columns

    # Verify the OCR stage output
    ocr_result = result["text"][0].text
    assert len(ocr_result) > 0

    # Verify the draw stage output
    draw_result = result["image_with_boxes"][0]
    assert draw_result.exception == ""

    # Run the pipeline on the input PDF file and check for exceptions
    pdf_result = pipeline.fromFile(pdf_file)
    assert 'Unable to read image' in pdf_result["image_with_boxes"][0].exception

    # Restore the original UserDefinedFunction
    pyspark.sql.udf.UserDefinedFunction = temp
