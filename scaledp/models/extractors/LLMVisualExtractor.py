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
    retry_if_exception_type,
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
        "temperature": 1.0,
        "schemaByPrompt": True,
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

        @retry(
            retry=retry_if_exception_type(RateLimitError),
            wait=wait_random_exponential(min=1, max=self.getDelay()),
            stop=stop_after_attempt(self.getMaxRetry()),
        )
        def completion_with_backoff(**kwargs):
            logging.info(f"Calling LLM API")
            return client.beta.chat.completions.parse(**kwargs)

        kwargs = {}

        content = [{"type": "text", "text": params["prompt"]}]
        if self.getSchemaByPrompt():
            content.append(
                {
                    "type": "text",
                    "text": "Schema for the output json: "
                    + self.getSchema()
                    + " Always return valid json. Do not include schema to the output.",
                }
            )
            # kwargs["response_format"] = {"type": "json_object"}
        else:
            kwargs["response_format"] = self.getPaydanticSchema()

        for image in images:
            image_decoded = base64.b64encode(image.data).decode("utf-8")
            completion = completion_with_backoff(
                model=params["model"],
                messages=[
                    {
                        "role": "system",
                        "content": params["systemPrompt"],
                        "role": "user",
                        "content": content
                        + [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_decoded}"
                                },
                            }
                        ],
                    },
                ],
                temperature=self.getTemperature(),
                **kwargs,
            )

            result = (
                completion.choices[0]
                .message.content.replace("```json", "")
                .replace("```", "")
            )

            results.append(
                ExtractorOutput(
                    path=image.path,
                    # data=json.dumps(completion.choices[0].message.parsed.json(), indent=4, ensure_ascii=False),
                    # data=completion.choices[0].message.parsed.json(),
                    data=json.dumps(json.loads(result), indent=4, ensure_ascii=False),
                    type="LLMVisualExtractor",
                    exception="",
                )
            )
        return results
