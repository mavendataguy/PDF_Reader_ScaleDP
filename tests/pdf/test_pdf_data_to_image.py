from sparkpdf.models.recognizers.TesseractOcr import TesseractOcr
from sparkpdf.pdf.PdfDataToImage import PdfDataToImage


def test_pdf_data_to_text(pdf_df):
    # Initialize the PdfDataToImage stage with specific input and output columns
    pdf_data_to_image = PdfDataToImage(inputCol="content", outputCol="image")

    # Initialize the Tesseract OCR stage with specific input and output columns
    ocr = TesseractOcr(inputCol="image", outputCol="text")

    # Transform the PDF dataframe to image dataframe and then to text dataframe
    result = ocr.transform(pdf_data_to_image.transform(pdf_df)).collect()

    # Verify the pipeline result
    assert len(result) == 2, "Expected exactly two results"

    # Verify the presence of the 'text' field in the result
    assert hasattr(result[0], "text"), "Expected 'text' field in the result"

    # Verify that there is no exception in the OCR result
    assert result[0].text.exception == "", "Expected no exception in the OCR result"

    # Verify that the detected text contains the expected substring
    assert "UniDoc Medial Center" in result[0].text.text, "Expected 'UniDoc Medical Center' in the detected text"

def test_pdf_data_to_image_class(pdf_file):
    # Read the PDF file
    with open(pdf_file, "rb") as f:
        data = f.read()

    pdf_to_image = PdfDataToImage()

    # Transform the PDF data to images and verify the result
    result = list(pdf_to_image.transform_udf(data, "path"))
    assert len(result) == 2, "Expected 2 images from the PDF file"

    # Test with None input and verify the result
    result_none = list(pdf_to_image.transform_udf(None, "path"))
    assert len(result_none) == 0, "Expected 0 images from None input"
