import io
import logging
import base64
from pyspark import keyword_only

from scaledp.params import *
from scaledp.schemas.Box import Box
from scaledp.schemas.Document import Document
from scaledp.models.recognizers.BaseOcr import BaseOcr
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type
)


class LLMOcr(BaseOcr, HasLLM, HasPrompt):

    defaultParams = {
        "inputCol": "image",
        "outputCol": "text",
        "keepInputData": False,
        "scaleFactor": 1.0,
        "scoreThreshold": 0.5,
        "lang": ["eng"],
        "lineTolerance": 0,
        "keepFormatting": False,
        "partitionMap": False,
        "numPartitions": 0,
        "pageCol": "page",
        "pathCol": "path",
        "systemPrompt": "You are ocr.",
        "prompt": """Please extract text from the image.""",
        "model": "gemini-1.5-flash",
        "apiBase": None,
        "apiKey": None,
        "delay": 30,
        "maxRetry": 6,
        "propagateError": False,
    }

    @keyword_only
    def __init__(self, **kwargs):
        super(LLMOcr, self).__init__()
        self._setDefault(**self.defaultParams)
        self._set(**kwargs)

    @classmethod
    def call_ocr(cls, images, params):
        from openai import RateLimitError

        client = cls.getClient(params["apiKey"], params["apiBase"])
        results = []

        @retry(retry=retry_if_exception_type(RateLimitError),
               wait=wait_random_exponential(min=1,
                                            max=params["delay"]),
               stop=stop_after_attempt(params["maxRetry"]))
        def completion_with_backoff(**kwargs):
            logging.info(f"Calling LLM API")
            return client.beta.chat.completions.parse(**kwargs)

        for image, image_path in images:
            buff = io.BytesIO()
            image.save(buff, "png")
            image_decoded = base64.b64encode(buff.getvalue()).decode('utf-8')
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
                #response_format=self.getPaydanticSchema(),
            )

            results.append(
                Document(path=image_path, text=completion.choices[0].message.content, type="text", bboxes=[])
            )
        return results
