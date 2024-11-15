import json
from pyspark import keyword_only

from ...enums import Device
from sparkpdf.params import *
from sparkpdf.schemas.Image import Image
from sparkpdf.schemas.Box import Box
from sparkpdf.schemas.Document import Document
from sparkpdf.models.recognizers.BaseOcr import BaseOcr
import pandas as pd

class SuryaOcr(BaseOcr, HasDevice, HasBatchSize):

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
    }

    @keyword_only
    def __init__(self, **kwargs):
        super(SuryaOcr, self).__init__()
        self._setDefault(**self.defaultParams)
        self._set(**kwargs)

    def call_ocr(self, image, scale_factor, image_path):
        from surya.ocr import run_ocr
        from surya.model.detection.model import load_model as load_det_model, load_processor as load_det_processor
        from surya.model.recognition.model import load_model as load_rec_model
        from surya.model.recognition.processor import load_processor as load_rec_processor
        from surya.settings import settings

        if self.getDevice() == Device.CPU.value:
            device = "cpu"
        else:
            device = "cuda"

        langs = self.getLang()

        settings.DETECTOR_BATCH_SIZE = self.getBatchSize()
        settings.RECOGNITION_BATCH_SIZE = self.getBatchSize()

        det_processor, det_model = load_det_processor(), load_det_model(device=device)
        rec_model, rec_processor = load_rec_model(device=device), load_rec_processor()

        predictions = run_ocr([image], [langs], det_model, det_processor, rec_model, rec_processor)

        boxes = [ Box(text=x.text, score=x.confidence, x=x.bbox[0],
                      y=x.bbox[1], width=x.bbox[2] - x.bbox[0], height=x.bbox[3] - x.bbox[1])
                        .toString().scale(1 / scale_factor) for x in predictions[0].text_lines]

        if self.getKeepFormatting():
            text = SuryaOcr.box_to_formatted_text(boxes, self.getLineTolerance())
        else:
            text = "\n".join([str(w.text) for w in boxes])


        return Document(path=image_path,
                        text=text,
                        type="text",
                        bboxes=boxes)

    @staticmethod
    def transform_udf_pandas(images: pd.DataFrame, params: pd.Series) -> pd.DataFrame:
        params = json.loads(params[0])
        from surya.ocr import run_ocr
        from surya.model.detection.model import load_model as load_det_model, load_processor as load_det_processor
        from surya.model.recognition.model import load_model as load_rec_model
        from surya.model.recognition.processor import load_processor as load_rec_processor
        from surya.settings import settings

        if int(params['device']) == Device.CPU.value:
            device = "cpu"
        else:
            device = "cuda"

        langs = params["lang"]

        settings.DETECTOR_BATCH_SIZE = params["batchSize"]
        settings.RECOGNITION_BATCH_SIZE = params["batchSize"]

        det_processor, det_model = load_det_processor(), load_det_model(device=device)
        rec_model, rec_processor = load_rec_model(device=device), load_rec_processor()
        results = []
        resized_images = []
        for index, image in images.iterrows():
            if not isinstance(image, Image):
                image = Image(**image.to_dict())
            image_pil = image.to_pil()
            scale_factor = params['scaleFactor']
            if scale_factor != 1.0:
                resized_image = image_pil.resize(
                    (int(image_pil.width * scale_factor), int(image_pil.height * scale_factor)))
            else:
                resized_image = image_pil
            resized_images.append(resized_image)

        predictions = run_ocr(resized_images, [langs]* len(resized_images), det_model, det_processor, rec_model, rec_processor)

        for prediction in predictions:
            boxes = [Box(text=x.text, score=x.confidence, x=x.bbox[0],
                         y=x.bbox[1], width=x.bbox[2] - x.bbox[0], height=x.bbox[3] - x.bbox[1])
                     .toString().scale(1 / scale_factor) for x in prediction.text_lines]

            if params["keepFormatting"]:
                text = SuryaOcr.box_to_formatted_text(boxes, params["lineTolerance"])
            else:
                text = "\n".join([str(w.text) for w in boxes])

            results.append(Document(path=image.path,
                                    text=text,
                                    type="text",
                                    bboxes=boxes))

        det_model.cpu()
        rec_model.cpu()
        del det_model, rec_model

        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()

        return pd.DataFrame(results)
