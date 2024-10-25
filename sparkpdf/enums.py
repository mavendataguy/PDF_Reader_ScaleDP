from enum import IntEnum, Enum


class ImageType(Enum):
    FILE = "file"
    OPENCV = "opencv"
    PIL = "pil"
    WEBP = "webp"


class Device(IntEnum):
    CPU = -1
    CUDA = 0
    CUDA_0 = 0
    CUDA_1 = 1
    CUDA_2 = 2
