from dataclasses import dataclass
from scaledp.utils.dataclass import (
    map_dataclass_to_struct,
    register_type,
    apply_nullability,
)
from scaledp.schemas.Box import Box


@dataclass(order=True)
class Entity:
    entity_group: str
    score: float
    word: str
    start: int
    end: int
    boxes: list[Box]

    @staticmethod
    def get_schema():
        return apply_nullability(map_dataclass_to_struct(Entity), True)


register_type(Entity, Entity.get_schema)
