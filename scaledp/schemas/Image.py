from pyspark.sql.types import *
import imagesize
import io
import logging
from PIL import Image as pImage
import traceback
from dataclasses import dataclass
from ..enums import ImageType
from scaledp.utils.dataclass import map_dataclass_to_struct, register_type, BinaryT


@dataclass(order=True)
class Image(object):
    """
    Image object for represent image data in Spark Dataframe
    """

    path: str
    resolution: int = 0
    data: BinaryT = bytes()
    imageType: str = ImageType.FILE.value
    exception: str = ""
    height: int = 0
    width: int = 0

    def to_pil(self):
        if self.imageType == ImageType.FILE.value or self.imageType == ImageType.WEBP.value:
            return pImage.open(io.BytesIO(self.data))

    def to_io_stream(self):
        return io.BytesIO(self.data)

    def to_opencv(self):
        if self.imageType == ImageType.FILE.value:
            return pImage.open(io.BytesIO(self.data))

    def to_webp(self):
        if self.imageType == ImageType.FILE.value:
            image = pImage.open(io.BytesIO(self.data))
            buff = io.BytesIO()
            image.save(buff, "webp")
            self.data = buff.getvalue()
        return self

    @staticmethod
    def from_binary(data, path, imageType, resolution=None, width=None, height=None):
        img = Image(path=path, data=data, imageType=ImageType.FILE.value, resolution=resolution)
        if data is None or len(data) == 0:
            raise ValueError("Empty image data.")
        if imageType in (ImageType.FILE.value, ImageType.WEBP.value):
            if height is not None:
                img.height = height
            if width is not None:
                img.width = width
            if width is None and height is None:
                img.width, img.height = imagesize.get(io.BytesIO(img.data))
                if img.width == -1:
                    raise Exception("Unable to read image.")
                logging.info(f"Image size: {img.width}x{img.height}")
        return img

    @staticmethod
    def from_pil(data, path, imageType, resolution):
        buff = io.BytesIO()
        if imageType == ImageType.WEBP.value:
            data.save(buff, "webp")
        else:
            data.save(buff, "png")
        img = Image(
            path=path,
            data=buff.getvalue(),
            imageType=ImageType.FILE.value,
            width=data.width,
            height=data.height,
            resolution=resolution,
        )
        return img

    @staticmethod
    def get_schema():
        return map_dataclass_to_struct(Image)

    def __str__(self):
        return f"Image(path={self.path}, resolution={self.resolution}, imageType={self.imageType}, exception={self.exception}, height={self.height}, width={self.width})"

    def __repr__(self):
        return self.__str__()


register_type(Image, Image.get_schema)
