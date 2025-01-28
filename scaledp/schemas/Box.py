from dataclasses import dataclass
from typing import Any, Dict

from scaledp.utils.dataclass import map_dataclass_to_struct, register_type


@dataclass(order=True)
class Box:
    """Box object for represent bounding box data in Spark Dataframe."""

    text: str
    score: float
    x: int
    y: int
    width: int
    height: int

    def to_string(self) -> "Box":
        self.text = str(self.text)
        return self

    def json(self) -> Dict[str, Any]:
        return {"text": self.text}

    @staticmethod
    def get_schema():
        return map_dataclass_to_struct(Box)

    def scale(self, factor: float, padding: int = 0) -> "Box":
        return Box(
            text=self.text,
            score=self.score,
            x=int(self.x * factor) - padding,
            y=int(self.y * factor) - padding,
            width=int(self.width * factor) + padding,
            height=int(self.height * factor) + padding,
        )

    def shape(self, padding: int = 0) -> list[tuple[int, int]]:
        return [
            (self.x - padding, self.y - padding),
            (self.x + self.width + padding, self.y + self.height + padding),
        ]

    def bbox(self, padding: int = 0) -> list[int]:
        return [
            self.x - padding,
            self.y - padding,
            self.x + self.width + padding,
            self.y + self.height + padding,
        ]

    @staticmethod
    def from_bbox(box: list[int], label: str = "", score: float = 0):
        return Box(
            text=label,
            score=float(score),
            x=int(box[0]),
            y=int(box[1]),
            width=int(box[2] - box[0]),
            height=int(box[3] - box[1]),
        )


register_type(Box, Box.get_schema)
