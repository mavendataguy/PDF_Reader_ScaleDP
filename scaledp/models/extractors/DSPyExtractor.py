import json
import os

from .BaseExtractor import BaseExtractor
from pyspark import keyword_only

from ...params import HasLLM, HasSchema
from ...schemas.ExtractorOutput import ExtractorOutput
from ...utils.pydantic_shema_utils import json_schema_to_model


class DSPyExtractor(BaseExtractor, HasLLM, HasSchema):

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
    }

    @keyword_only
    def __init__(self, **kwargs):
        super(DSPyExtractor, self).__init__()
        self._setDefault(**self.defaultParams)
        self._set(**kwargs)
        self.pipeline = None

    def call_extractor(self, documents, params):
        import dspy
        lm = dspy.LM(
            params["model"],
            api_base=params["apiBase"] or os.environ["OPENAI_BASE_URL"],
            api_key=params["apiKey"] or os.environ["OPENAI_API_KEY"],
        )
        dspy.configure(lm=lm)

        schema = json.loads(params["schema"])
        schema = json_schema_to_model(schema, schema.get("$defs", {}))

        class ExtractData(dspy.Signature):
            """improve a recognized text and Extract structured information from the receipt."""

            text: str = dspy.InputField(
                desc="""OCR recognized text of the receipt from the Ukrainian store""",
            )
            data: schema = dspy.OutputField(
                desc="Structured data from the receipt with fixes and improvements of OCR recognition."
            )

        module = dspy.ChainOfThought(ExtractData)

        results = []
        for document in documents:
            data = module(text=document.text).data
            results.append(
                ExtractorOutput(
                    path=document.path,
                    data=data.json(),
                    type="DSPyExtractor",
                    exception="",
                )
            )
        return results
