import datetime
import logging
from typing import List

from pydantic import BaseModel
from pyspark import keyword_only
from pyspark.ml.param import Param, Params, TypeConverters

from scaledp.params import HasLLM, HasPrompt, HasPropagateError
from scaledp.models.ner.BaseNer import BaseNer
from ...enums import Device
from scaledp.schemas.Entity import Entity
from scaledp.schemas.NerOutput import NerOutput
import pandas as pd
import json

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type,
)


def log_info(msg):
    print(f"{datetime.datetime.now()} INFO: {msg}")


class LLMNer(BaseNer, HasLLM, HasPrompt, HasPropagateError):

    tags = Param(
        Params._dummy(),
        "tags",
        "Ner tags.",
        typeConverter=TypeConverters.toListString,
    )

    defaultParams = {
        "inputCol": "text",
        "outputCol": "ner",
        "keepInputData": True,
        "model": "d4data/biomedical-ner-all",
        "whiteList": [],
        "numPartitions": 1,
        "device": Device.CPU,
        "batchSize": 1,
        "scoreThreshold": 0.0,
        "pageCol": "page",
        "pathCol": "path",
        "systemPrompt": "Please extract following NER Tags from the text as json: ",
        "prompt": """Please extract text from the image.""",
        "model": "gemini-1.5-flash-8b",
        "apiBase": "",
        "apiKey": "",
        "delay": 30,
        "maxRetry": 6,
        "propagateError": False,
        "tags": ["PERSON", "LOCATION", "DATE", "EMAIL", "PHONE", "ORGANIZATION", "ID"],
        "partitionMap": False,
    }

    def getPaydanticSchema(self):

        class Entity(BaseModel):
            entity_group: str
            score: float
            word: str
            start: int
            end: int

        class NerOutput(BaseModel):
            entities: List[Entity]

        return NerOutput

    @keyword_only
    def __init__(self, **kwargs):
        super(LLMNer, self).__init__()
        self._setDefault(**self.defaultParams)
        self._set(**kwargs)
        self.pipeline = None

    def transform_udf(self, document, params=None):

        if params is None:
            params = self.get_params()
        params = json.loads(params)

        mapping = []
        for idx, box in enumerate(document.bboxes):
            mapping.extend([idx] * (len(box.text) + 1))

        from openai import RateLimitError

        client = self.getClient(params["apiKey"], params["apiBase"])
        results = []

        @retry(
            retry=retry_if_exception_type(RateLimitError),
            wait=wait_random_exponential(min=1, max=params["delay"]),
            stop=stop_after_attempt(params["maxRetry"]),
        )
        def completion_with_backoff(**kwargs):
            logging.info(f"Calling LLM API")
            return client.beta.chat.completions.parse(**kwargs)

        schema = self.getPaydanticSchema().model_json_schema()

        completion = completion_with_backoff(
            model=params["model"],
            messages=[
                {
                    "role": "user",
                    "content": f'Pleas extract NER tags: {",".join(params["tags"])} as json with schema: {schema}. From the text:'
                    + document.text,
                },
            ],
            response_format={"type": "json_object"},
        )
        entities = []

        data = json.loads(
            completion.choices[0]
            .message.content.replace("```json", "")
            .replace("```", "")
        )
        for tag in data["entities"]:
            if (
                len(self.getWhiteList()) > 0
                and tag["entity_group"] not in self.getWhiteList()
            ):
                continue
            boxes = []
            word = tag["word"]
            for idx, box in enumerate(document.bboxes):
                if any(
                    word.lower() in box.text.lower()
                    and (len(word) > 2 or abs(len(word) - len(box.text)) < 2)
                    for word in word.split(" ")
                    if len(word) > 1
                ):
                    boxes.append(box)
            t = Entity(
                entity_group=tag["entity_group"],
                score=float(tag["score"]),
                word=tag["word"],
                start=tag["start"],
                end=tag["end"],
                boxes=boxes,
            )
            entities.append(t)

        output = NerOutput(path=document.path, entities=entities, exception="")
        return output

    @staticmethod
    def transform_udf_pandas(
        texts: pd.DataFrame, params: pd.Series
    ) -> pd.DataFrame:  # pragma: no cover
        params = json.loads(params[0])
        model = params["model"]

        results = []

        return pd.DataFrame(results)
