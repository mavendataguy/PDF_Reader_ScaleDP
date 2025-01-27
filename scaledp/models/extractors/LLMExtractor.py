import json
import time
import logging

from .BaseExtractor import BaseExtractor
from pyspark import keyword_only

from ...params import HasLLM, HasSchema, HasPrompt
from ...schemas.ExtractorOutput import ExtractorOutput
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type,
)


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
        "systemPrompt": "You are data extractor from the scanned images.",
        "delay": 30,
        "maxRetry": 6,
        "propagateError": False,
        "temperature": 1.0,
        "schemaByPrompt": True,
    }

    @keyword_only
    def __init__(self, **kwargs):
        super(LLMExtractor, self).__init__()
        self._setDefault(**self.defaultParams)
        self._set(**kwargs)
        self.pipeline = None

    def call_extractor(self, documents, params):
        from openai import RateLimitError

        client = self.getOIClient()

        @retry(
            retry=retry_if_exception_type(RateLimitError),
            wait=wait_random_exponential(min=1, max=self.getDelay()),
            stop=stop_after_attempt(self.getMaxRetry()),
        )
        def completion_with_backoff(**kwargs):
            logging.info(f"Calling LLM API")
            return client.beta.chat.completions.parse(**kwargs)

        results = []
        for document in documents:
            data = document.text

            content = [{"type": "text", "text": params["prompt"]}]
            kwargs = {}

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

            completion = completion_with_backoff(
                model=params["model"],
                messages=[
                    {
                        # "role": "system",
                        # "content": params["systemPrompt"],
                        "role": "user",
                        "content": [
                            {"type": "text", "text": data},
                        ],
                    },
                    {"role": "user", "content": content},
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
                    path=document.path,
                    # data=json.dumps(completion.choices[0].message.parsed.json(),indent=4, ensure_ascii=False),
                    # completion.choices[0].message.parsed.json()
                    data=json.dumps(json.loads(result), indent=4, ensure_ascii=False),
                    type="LLMExtractor",
                    exception="",
                )
            )
        return results
