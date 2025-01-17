import io
import pytest
from scaledp import ImageType
from scaledp.schemas.Image import Image

def test_image(image_pil_1x1):
    # Test with invalid image data
    with pytest.raises(ValueError):
        img = Image.from_binary(data=None, path='path', imageType=ImageType.FILE.value)

    image = Image.from_pil(image_pil_1x1, 'path', ImageType.FILE.value, 300)
    # Test to_webp method
    webp_image = image.to_webp()
    assert webp_image.imageType == ImageType.FILE.value
    assert webp_image.data is not None

    # Test to_pil method
    pil_image = image.to_pil()
    assert pil_image is not None
    assert pil_image.size == (1, 1)

    # Test to_io_stream method
    io_stream = image.to_io_stream()
    assert io_stream is not None
    assert isinstance(io_stream, io.BytesIO)

    # Test to_opencv method
    opencv_image = image.to_opencv()
    assert opencv_image is not None
    assert opencv_image.size == (1, 1)