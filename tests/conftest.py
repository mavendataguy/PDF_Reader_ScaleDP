import pytest

from sparkpdf.image.DataToImage import DataToImage
from sparkpdf.enums import ImageType

@pytest.fixture
def raw_image_df(spark_session):
    return spark_session.read.format("binaryFile").load("../sparkpdf/resources/images/InvoiceforMedicalRecords_10_722.png")

def image_df(spark_session):
    df = spark_session.read.format("binaryFile").load("../sparkpdf/resources/images/InvoiceforMedicalRecords_10_722.png")
    bin_to_image = DataToImage().setImageType(ImageType.WEBP.value)
    return bin_to_image.transform(df)
