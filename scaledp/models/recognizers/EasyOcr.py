import gc
from pyspark import keyword_only
import numpy as np

from scaledp.enums import Device
from scaledp.params import *
from scaledp.schemas.Box import Box
from scaledp.schemas.Document import Document
from scaledp.models.recognizers.BaseOcr import BaseOcr


class EasyOcr(BaseOcr, HasDevice, HasBatchSize):

    defaultParams = {
        "inputCol": "image",
        "outputCol": "text",
        "keepInputData": False,
        "scaleFactor": 1.0,
        "scoreThreshold": 0.5,
        "lang": ["eng"],
        "lineTolerance": 0,
        "keepFormatting": False,
        "partitionMap": False,
        "numPartitions": 0,
        "pageCol": "page",
        "pathCol": "path",
        "device": Device.CPU,
        "batchSize": 2,
        "propagateError": False,
    }

    @keyword_only
    def __init__(self, **kwargs):
        super(EasyOcr, self).__init__()
        self._setDefault(**self.defaultParams)
        self._set(**kwargs)

    @staticmethod
    def points_to_box(points, text, score):
        """
        Convert a set of four corner points to (x, y, width, height).

        Args:
            points (list of tuple): List of four (x, y) tuples representing the corners.

        Returns:
            tuple: A tuple (x, y, width, height).
        """
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]

        x = min(x_coords)
        y = min(y_coords)
        width = max(x_coords) - x
        height = max(y_coords) - y

        return Box(text=text, score=score, x=x, y=y, width=width, height=height)

    @classmethod
    def call_ocr(cls, images, params):
        import easyocr
        import torch

        if int(params["device"]) == Device.CPU.value:
            device = False
        else:
            device = True

        langs = params["lang"]
        scale_factor = params["scaleFactor"]
        reader = easyocr.Reader(langs, device)
        results = []
        for image, image_path in images:

            image = np.array(image.convert("RGB"))[:, :, ::-1].copy()
            result = reader.readtext(image)
            boxes = [
                EasyOcr.points_to_box(box, text, float(score)).toString().scale(1 / scale_factor)
                for box, text, score in result
            ]

            if params["keepFormatting"]:
                text = EasyOcr.box_to_formatted_text(boxes, params["lineTolerance"])
            else:
                text = "\n".join([str(w.text) for w in boxes])

            results.append(Document(path=image_path, text=text, type="text", bboxes=boxes))

        gc.collect()
        if int(params["device"]) == Device.CUDA.value:
            torch.cuda.empty_cache()

        return results
