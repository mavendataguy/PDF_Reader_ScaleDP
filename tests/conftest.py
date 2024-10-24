import pytest

from sparkpdf.image.DataToImage import DataToImage
from sparkpdf.enums import ImageType

@pytest.fixture
def raw_image_df(spark_session):
    return spark_session.read.format("binaryFile").load("../sparkpdf/resources/images/InvoiceforMedicalRecords_10_722.png")


@pytest.fixture
def binary_pdf_df(spark_session):
    df = spark_session.read.format("binaryFile").load(
        "../sparkpdf/resources/pdfs/unipdf-medical-bill.pdf")
    return df

@pytest.fixture
def pdf_df(spark_session):
    df = spark_session.read.format("binaryFile").load(
        "../sparkpdf/resources/pdfs/unipdf-medical-bill.pdf")
    return df

@pytest.fixture
def image_df(spark_session):
    df = spark_session.read.format("binaryFile").load("../sparkpdf/resources/images/InvoiceforMedicalRecords_10_722.png")
    bin_to_image = DataToImage().setImageType(ImageType.WEBP.value)
    return bin_to_image.transform(df)
