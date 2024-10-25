from dataclasses import dataclass
from pyspark_types.dataclass import map_dataclass_to_struct, register_type
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

    def shape(self):
        return [(self.x, self.y), (self.x + self.width, self.y + self.height)]

    def bbox(self):
        return [self.x, self.y, self.x + self.width, self.y + self.height]


register_type(Box, Box.get_schema)
