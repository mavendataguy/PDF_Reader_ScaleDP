# Description: Schema Document
from sparkpdf.utils.dataclass import map_dataclass_to_struct, register_type
from dataclasses import dataclass
from sparkpdf.schemas.Box import Box

@dataclass(order=True)
class Document:
    path: str
    text: str
    type: str
    bboxes: list[Box]
    exception: str = ""

    @staticmethod
    def get_schema():
        return map_dataclass_to_struct(Document)

register_type(Document, Document.get_schema)
