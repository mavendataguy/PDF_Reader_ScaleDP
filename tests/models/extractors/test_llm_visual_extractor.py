from datetime import date, time

from scaledp.models.extractors.LLMVisualExtractor import LLMVisualExtractor
from pydantic import BaseModel, Field, field_validator
import json
from scaledp.metrics.object_similarity import calculate_similarity


class ReceiptItem(BaseModel):
    """Purchased items."""
    name: str
    quantity: float
    price_per_unit: float
    price: float
    product_code: str = Field(description="Product identifier (13 digits)")


class Address(BaseModel):
    """Address."""
    country_code: str
    state: str
    city: str
    street: str
    house: str


class ReceiptSchema(BaseModel):
    """Receipt."""
    company_name: str
    shop_name: str
    address: Address
    tax_id: str
    transaction_date: date = Field(description="Date of the transaction")
    transaction_time: time = Field(description="Time of the transaction")
    total_amount: float
    items: list[ReceiptItem]


def test_llm_visual_extractor(image_receipt_df, receipt_json, receipt_json_path):

    extractor = LLMVisualExtractor(model="gemini-1.5-flash", schema=ReceiptSchema)

    # Transform the image dataframe through the OCR and NER stages
    result_df = extractor.transform(image_receipt_df)

    # Cache the result for performance
    result = result_df.select("data").cache()

    # Collect the results
    data = result.collect()

    # Assert that there is exactly one result
    assert len(data) == 1

    # Assert that the 'data' field is present in the result
    assert hasattr(data[0], "data")

    # Check that exceptions is empty
    assert data[0].data.exception == ""
    receipt = ReceiptSchema.model_validate_json(data[0].data.data)
    if not receipt_json_path.exists():
        with receipt_json_path.open("w") as f:
            f.write(json.dumps(json.loads(data[0].data.data), indent=4, ensure_ascii=False))
    true_receipt = ReceiptSchema.model_validate_json(receipt_json)
    assert calculate_similarity(receipt, true_receipt) > 0.7

    result.select("data.data").show(1, False)

