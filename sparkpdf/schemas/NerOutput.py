
from dataclasses import dataclass
from sparkpdf.utils.dataclass import map_dataclass_to_struct, register_type
from sparkpdf.schemas.Entity import Entity


@dataclass(order=True)
class NerOutput:
    path: str
    entities: list[Entity]
    exception: str
    json: str = None
    @staticmethod
    def get_schema():
        return map_dataclass_to_struct(NerOutput)

register_type(NerOutput, NerOutput.get_schema)
