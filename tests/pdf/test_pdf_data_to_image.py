from sparkpdf.models.recognizers.TesseractOcr import TesseractOcr
from sparkpdf.pdf.PdfDataToImage import PdfDataToImage


def test_pdf_data_to_text(pdf_df):
    pdf_data_to_image = PdfDataToImage(inputCol="content", outputCol="image")
    ocr = TesseractOcr(inputCol="image", outputCol="text")
    result = ocr.transform(pdf_data_to_image.transform(pdf_df)).collect()
    assert (len(result) == 2)
    assert (hasattr(result[0], "text"))
    assert (result[0].text.exception == "")
    assert ("UniDoc Medial Center" in result[0].text.text)
