# Description: Schema for OCR output
from pyspark_types.dataclass import map_dataclass_to_struct, register_type
from dataclasses import dataclass
from sparkpdf.schemas.Box import Box

@dataclass(order=True)
class OcrOutput:
    path: str
    text: str
    type: str
    bboxes: list[Box]
    exception: str = ""

    @staticmethod
    def get_schema():
        return map_dataclass_to_struct(OcrOutput)

register_type(OcrOutput, OcrOutput.get_schema)
