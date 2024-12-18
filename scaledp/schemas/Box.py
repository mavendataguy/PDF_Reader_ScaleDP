from dataclasses import dataclass
from scaledp.utils.dataclass import map_dataclass_to_struct, register_type
@dataclass(order=True)
class Box:
    text: str
    score: float
    x: int
    y: int
    width: int
    height: int

    def toString(self):
        self.text = str(self.text)
        return self

    def json(self):
        return {"text": self.text }

    @staticmethod
    def get_schema():
        return map_dataclass_to_struct(Box)

    def scale(self, factor):
        return Box(text=self.text,
                   score=self.score,
                   x = int(self.x * factor),
                   y = int(self.y * factor),
                   width = int(self.width * factor),
                   height = int(self.height * factor))

    def shape(self, padding=0):
        return [(self.x - padding, self.y - padding), (self.x + self.width + padding, self.y + self.height + padding)]

    def bbox(self, padding=0):
        return [self.x - padding, self.y - padding, self.x + self.width + padding, self.y + self.height + padding]

    @staticmethod
    def fromBBox(box, label="", score=0):
        return Box(text=label,
                   score=float(score),
                   x=int(box[0]),
                   y=int(box[1]),
                   width=int(box[2] - box[0]),
                   height=int(box[3] - box[1]))


register_type(Box, Box.get_schema)
