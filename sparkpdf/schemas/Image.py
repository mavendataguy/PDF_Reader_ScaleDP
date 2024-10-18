from pyspark.sql.types import *
from ..enums import ImageType
import io
import logging
from PIL import Image as pImage
import traceback


class Image(object):
    """
    Image object for represent image data in Spark Dataframe
    """
    def __init__(self, origin, imageType=ImageType.FILE.value, data=bytes(), height=0, width=0, resolution=0, exception = ""):
        self.origin = origin
        self.height = height
        self.width = width
        self.resolution = resolution
        self.data = data
        self.imageType = imageType
        self.exception = exception

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
            i = pImage.open(io.BytesIO(self.data))
            buff = io.BytesIO()
            i.save(buff, "webp")
            self.data = buff.getvalue()
        return self

    @staticmethod
    def from_binary(data, origin, imageType, resolution=None, width=None, height=None):
        img = Image(origin=origin, data=data, imageType=ImageType.FILE.value, resolution=resolution)
        try:
            if data is None or len(data) == 0:
                raise Exception("Empty image data.")
            if imageType == ImageType.PIL.value:
                return img.toPIL()
            elif imageType == ImageType.OPENCV.value:
                return img.toOpecv()
            else:
                if height is not None:
                    img.height = height
                if width is not None:
                    img.width = width
                if width is None and height is None:
                    import imagesize
                    img.width, img.height = imagesize.get(io.BytesIO(img.data))
                return img
        except Exception as e:
            exception = traceback.format_exc()
            exception = f"ToImage: {exception}"
            logging.error(f"ToImage: Error in image extraction. {exception}")
            img.exception = exception
            return img

    @staticmethod
    def from_pil(data, origin, imageType, resolution):
        buff = io.BytesIO()
        if imageType == ImageType.WEBP.value:
            data.save(buff, "webp")
        else:
            data.save(buff, "png")
        img = Image(origin=origin, data=buff.getvalue(), imageType=ImageType.FILE.value, width=data.width, height=data.height, resolution=resolution)
        if imageType == ImageType.PIL.value:
            return img.toPIL()
        elif imageType == ImageType.OPENCV.value:
            return img.toOpecv()
        else:
            return img

    @staticmethod
    def get_schema():
        image_fields = ["origin", "imageType", "height", "width", "resolution", "data", "exception"]
        return StructType([
            StructField(image_fields[0], StringType(), True),
            StructField(image_fields[1], StringType(), True),
            StructField(image_fields[2], IntegerType(), False),
            StructField(image_fields[3], IntegerType(), False),
            StructField(image_fields[4], IntegerType(), False),
            StructField(image_fields[5], BinaryType(), True),
            StructField(image_fields[6], StringType(), True)])
