import json
import time

from .BaseExtractor import BaseExtractor
from pyspark import keyword_only

from ...params import HasLLM, HasSchema, HasPrompt
from ...schemas.ExtractorOutput import ExtractorOutput


class LLMExtractor(BaseExtractor, HasLLM, HasSchema, HasPrompt):

    defaultParams = {
        "inputCol": "text",
        "outputCol": "data",
        "keepInputData": True,
        "model": "llama3-8b-8192",
        "apiBase": None,
        "apiKey": None,
        "numPartitions": 1,
        "pageCol": "page",
        "pathCol": "path",
        "prompt": """Please extract data from the text as json.""",
        "delay": 0,
    }

    @keyword_only
    def __init__(self, **kwargs):
        super(LLMExtractor, self).__init__()
        self._setDefault(**self.defaultParams)
        self._set(**kwargs)
        self.pipeline = None

    def call_extractor(self, documents, params):
        client = self.getOIClient()
        results = []
        for document in documents:
            data = document.text
            completion = client.beta.chat.completions.parse(
                model=params["model"],
                messages=[
                    {
                        "role": "system",
                        "content": params["prompt"],
                        "role": "user",
                        "content": data
                    },
                ],
                response_format=self.getPaydanticSchema(),
            )
            if self.getDelay():
                time.sleep(self.getDelay())
            results.append(
                ExtractorOutput(
                    path=document.path,
                    #data=json.dumps(completion.choices[0].message.parsed.json(),indent=4, ensure_ascii=False),
                    data=completion.choices[0].message.parsed.json(),
                    type="LLMExtractor",
                    exception="",
                )
            )
        return results
