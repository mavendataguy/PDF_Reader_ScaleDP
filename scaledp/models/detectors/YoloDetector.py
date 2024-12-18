from pyspark import keyword_only

from ...enums import Device
from scaledp.models.detectors.BaseDetector import BaseDetector
from scaledp.params import HasDevice, HasBatchSize
import gc

from scaledp.schemas.Box import Box
from scaledp.schemas.DetectorOutput import DetectorOutput


class YoloDetector(BaseDetector, HasDevice, HasBatchSize):

    defaultParams = {
        "inputCol": "image",
        "outputCol": "boxes",
        "keepInputData": False,
        "scaleFactor": 1.0,
        "scoreThreshold": 0.5,
        "device": Device.CPU,
        "batchSize": 2,
    }

    @keyword_only
    def __init__(self, **kwargs):
        super(YoloDetector, self).__init__()
        self._setDefault(**self.defaultParams)
        self._set(**kwargs)

    @classmethod
    def call_detector(cls, images, params):
        from ultralytics import YOLO
        import torch
        detector = YOLO(params['model'])

        if int(params['device']) == Device.CPU.value:
            device = "cpu"
        else:
            device = "cuda"

        results = detector.to(device)(
            [image[0] for image in images],
            conf=params["scoreThreshold"],
            save_conf=True,
        )

        results_final = []
        for res, (image, image_path) in zip(results, images):
            boxes = []
            for box in res.boxes:
                boxes.append(Box.fromBBox(box.xyxy[0]))
            results_final.append(DetectorOutput(path=image_path,
                                          type="yolo",
                                          bboxes=boxes))

        gc.collect()
        if int(params['device']) == Device.CUDA.value:
            torch.cuda.empty_cache()

        return results_final
