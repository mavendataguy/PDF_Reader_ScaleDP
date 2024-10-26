import pytest

from sparkpdf.image.DataToImage import DataToImage
from sparkpdf.enums import ImageType

@pytest.fixture
def raw_image_df(spark_session, resource_path_root):
    return spark_session.read.format("binaryFile").load(
        (resource_path_root / "images/InvoiceforMedicalRecords_10_722.png").absolute().as_posix())


@pytest.fixture
def binary_pdf_df(spark_session, resource_path_root):
    df = spark_session.read.format("binaryFile").load(
        (resource_path_root / "pdfs/unipdf-medical-bill.pdf").absolute().as_posix())
    return df

@pytest.fixture
def pdf_df(spark_session, resource_path_root):
    df = spark_session.read.format("binaryFile").load(
        (resource_path_root / "pdfs/unipdf-medical-bill.pdf").absolute().as_posix())
    return df

@pytest.fixture
def image_df(spark_session, resource_path_root):
    df = spark_session.read.format("binaryFile").load(
        (resource_path_root / "images/InvoiceforMedicalRecords_10_722.png").absolute().as_posix())
    bin_to_image = DataToImage().setImageType(ImageType.WEBP.value)
    return bin_to_image.transform(df)
