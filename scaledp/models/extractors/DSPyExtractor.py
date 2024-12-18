import json

import pydantic
from pyspark.ml.param import Param, Params, TypeConverters

from .BaseExtractor import BaseExtractor
from ...enums import Device
from pyspark import keyword_only

from ...params import HasLLM
from ...schemas.ExtractorOutput import ExtractorOutput
from ...utils.pydantic_shema_utils import json_schema_to_model


class DSPyExtractor(BaseExtractor, HasLLM):

    schema = Param(Params._dummy(), "schema",
                  "Output schema.",
                  typeConverter=TypeConverters.toString)

    defaultParams = {
        "inputCol": "text",
        "outputCol": "data",
        "keepInputData": True,
        "model": "llama3-8b-8192",
        "apiBase": "https://api.groq.com/openai/v1",
        "apiKey": "",
        "numPartitions": 1,
        "pageCol": "page",
        "pathCol": "path"
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
            params['model'],
            api_base=params['apiBase'],
            api_key=params['apiKey'],
        )
        dspy.configure(lm=lm)

        schema = json.loads(params['schema'])
        schema = json_schema_to_model(schema, schema.get('$defs', {}))

        class ExtractData(dspy.Signature):
            """improve a recognized text and Extract structured information from the receipt."""

            text: str = dspy.InputField(
                desc="""text representation of the receipt""",
            )
            data: schema = dspy.OutputField(
                desc="improve a recognized text and extract a structured representation of the receipt",
            )


        module = dspy.ChainOfThought(ExtractData)


        results = []
        for document in documents:
            print(document.text)

            data = module(text=document.text).data
            results.append(ExtractorOutput(path=document.path,
                               data=data.json(),
                               type="DSPyExtractor",
                               exception=""))
        return results

    def getSchema(self):
        """
        Gets the value of schema or its default value.
        """
        return self.getOrDefault(self.schema)

    def setSchema(self, value):
        """
        Sets the value of :py:attr:`schema`.
        """
        return self._set(schema=value)