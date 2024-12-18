import pytest
from PIL import Image as pImage

from scaledp.image.DataToImage import DataToImage
from scaledp.enums import ImageType


@pytest.fixture
def image_file(resource_path_root):
    return (resource_path_root / "images/Invoice.png").absolute().as_posix()

@pytest.fixture
def receipt_file(resource_path_root):
    return (resource_path_root / "images" / "receipt.jpg").absolute().as_posix()

@pytest.fixture
def image_pil(image_file):
    return pImage.open(image_file)

@pytest.fixture
def image_pil_1x1():
    return pImage.new('RGB', (1, 1), color='red')

@pytest.fixture
def image(image_pil):
    from scaledp.schemas.Image import Image
    return Image.from_pil(image_pil, "test", ImageType.FILE.value, 300)

@pytest.fixture
def image_line(resource_path_root):
    from scaledp.schemas.Image import Image
    return Image.from_pil(pImage.open((resource_path_root / "images/text_line.png").absolute().as_posix()), "test", ImageType.FILE.value, 300)

@pytest.fixture
def raw_image_df(spark_session, resource_path_root):
    return spark_session.read.format("binaryFile").load(
        (resource_path_root / "images/Invoice.png").absolute().as_posix())


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
def pdf_file(resource_path_root):
    return (resource_path_root / "pdfs/unipdf-medical-bill.pdf").absolute().as_posix()

@pytest.fixture
def image_df(spark_session, resource_path_root):
    df = spark_session.read.format("binaryFile").load(
        (resource_path_root / "images/Invoice.png").absolute().as_posix())
    bin_to_image = DataToImage().setImageType(ImageType.WEBP.value)
    return bin_to_image.transform(df)

@pytest.fixture
def image_line_df(spark_session, resource_path_root):
    df = spark_session.read.format("binaryFile").load(
        (resource_path_root / "images/text_line.png").absolute().as_posix())
    bin_to_image = DataToImage().setImageType(ImageType.WEBP.value)
    return bin_to_image.transform(df)

@pytest.fixture
def image_receipt_df(spark_session, resource_path_root):
    df = spark_session.read.format("binaryFile").load(
        (resource_path_root / "images" / "receipt.jpg").absolute().as_posix())
    bin_to_image = DataToImage().setImageType(ImageType.WEBP.value)
    return bin_to_image.transform(df)

@pytest.fixture
def text_df(spark_session, resource_path_root):
    df = spark_session.read.text(
        (resource_path_root / "texts/example.txt").absolute().as_posix(), wholetext=True)
    return df
