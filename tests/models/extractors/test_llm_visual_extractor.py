from datetime import date, time

from scaledp.models.extractors.GeminiVisualExtractor import GeminiVisualExtractor
from scaledp.models.extractors.LLMVisualExtractor import LLMVisualExtractor
from scaledp.models.recognizers.TesseractOcr import TesseractOcr
from pydantic import BaseModel, Field
import json



class ReceiptItem(BaseModel):
    """Purchased items."""
    name: str
    quantity: float
    price_per_unit: float
    price: float
    sko: str = Field(description="Product identifier (13 digits)")


class ReceiptSchema(BaseModel):
    """Receipt."""
    company_name: str
    shop_name: str
    address: str
    tax_id: str
    transaction_date: date = Field(description="Date of the transaction")
    transaction_time: time = Field(description="Time of the transaction")
    total_amount: float
    items: list[ReceiptItem]


def test_llm_visual_extractor(image_receipt_df):

    extractor = LLMVisualExtractor(model="gemini-1.5-flash", schema=ReceiptSchema)

    # Transform the image dataframe through the OCR and NER stages
    result_df = extractor.transform(image_receipt_df)

    # Cache the result for performance
    result = result_df.select("data",).cache()

    # Collect the results
    data = result.collect()

    # Assert that there is exactly one result
    assert len(data) == 1

    # Assert that the 'data' field is present in the result
    assert hasattr(data[0], "data")

    # Check that exceptions is empty
    assert data[0].data.exception == ""

    result.select("data.data").show(1, False)
