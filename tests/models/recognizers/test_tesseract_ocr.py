from sparkpdf.image.DataToImage import DataToImage
from sparkpdf.models.recognizers.TesseractOcr import TesseractOcr

def test_tesseract_ocr(image_df):
    ocr = TesseractOcr()
    result = ocr.transform(image_df).collect()
    assert (len(result) == 1)
    # present text field
    assert (hasattr(result[0], "text"))
    # detected text
    assert ("Hospital: Dutton" in result[0].text.text)

def test_wrong_file_tesseract_ocr(pdf_df):
    ocr = TesseractOcr()
    result = ocr.transform(pdf_df).collect()
    assert (len(result) == 1)
    # present text field
    assert (hasattr(result[0], "text"))
    assert ("Unable to read image" in result[0].text.exception)
