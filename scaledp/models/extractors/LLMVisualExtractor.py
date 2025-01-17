import json
import time
import base64
import logging

from .BaseVisualExtractor import BaseVisualExtractor
from pyspark import keyword_only

from ...params import HasLLM, HasSchema, HasPrompt
from ...schemas.ExtractorOutput import ExtractorOutput
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type
)

class LLMVisualExtractor(BaseVisualExtractor, HasLLM, HasSchema, HasPrompt):

    defaultParams = {
        "inputCol": "image",
        "outputCol": "data",
        "keepInputData": True,
        "model": "gemini-1.5-flash",
        "apiBase": None,
        "apiKey": None,
        "numPartitions": 1,
        "pageCol": "page",
        "pathCol": "path",
        "prompt": """Please extract data from the scanned image as json. Date format is yyyy-mm-dd""",
        "systemPrompt": "You are data extractor from the scanned images.",
        "delay": 30,
        "maxRetry": 6,
        "propagateError": False,
    }

    @keyword_only
    def __init__(self, **kwargs):
        super(LLMVisualExtractor, self).__init__()
        self._setDefault(**self.defaultParams)
        self._set(**kwargs)
        self.pipeline = None

    def call_extractor(self, images, params):
        from openai import RateLimitError

        client = self.getOIClient()
        results = []


        @retry(retry=retry_if_exception_type(RateLimitError),
               wait=wait_random_exponential(min=1,
                                            max=self.getDelay()),
                                            stop=stop_after_attempt(self.getMaxRetry()))
        def completion_with_backoff(**kwargs):
            logging.info(f"Calling LLM API")
            return client.beta.chat.completions.parse(**kwargs)

        for image in images:
            image_decoded = base64.b64encode(image.data).decode('utf-8')
            completion = completion_with_backoff(
                model=params["model"],
                messages=[
                    {
                        "role": "system",
                        "content": params["systemPrompt"],
                        "role": "user",
                        "content": [
                            {
                              "type": "image_url",
                              "image_url": {
                                "url": f"data:image/jpeg;base64,{image_decoded}"
                              }
                            },
                            {
                              "type": "text",
                              "text": params["prompt"]
                            }
                          ]
                    },
                ],
                response_format=self.getPaydanticSchema(),
            )

            results.append(
                ExtractorOutput(
                    path=image.path,
                    #data=json.dumps(completion.choices[0].message.parsed.json(), indent=4, ensure_ascii=False),
                    data=completion.choices[0].message.parsed.json(),
                    type="LLMVisualExtractor",
                    exception="",
                )
            )
        return results
