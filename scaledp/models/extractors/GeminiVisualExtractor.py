import json

from .BaseVisualExtractor import BaseVisualExtractor
from pyspark import keyword_only

from ...params import HasLLM, HasSchema, HasPrompt
from ...schemas.ExtractorOutput import ExtractorOutput
from ...utils.pydantic_shema_utils import json_schema_to_model


class GeminiVisualExtractor(BaseVisualExtractor, HasLLM, HasSchema, HasPrompt):

    defaultParams = {
        "inputCol": "image",
        "outputCol": "data",
        "keepInputData": True,
        "model": "gemini-1.5-flash",
        "apiBase": "",
        "apiKey": "",
        "numPartitions": 1,
        "pageCol": "page",
        "pathCol": "path",
        "prompt": """Please extract data from the scanned image as json.""",
    }

    @keyword_only
    def __init__(self, **kwargs):
        super(GeminiVisualExtractor, self).__init__()
        self._setDefault(**self.defaultParams)
        self._set(**kwargs)
        self.pipeline = None

    def call_extractor(self, images, params):
        import google.generativeai as genai
        import base64
        import json

        schema = json.loads(params["schema"])
        schema = json_schema_to_model(schema, schema.get("$defs", {}))
        generation_config = {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
            "response_mime_type": "application/json",
            "response_schema": schema
        }

        genai.configure(api_key=params["apiKey"])

        model = genai.GenerativeModel(
            model_name=params["model"],
            generation_config=generation_config,
        )

        schema = json.loads(params["schema"])
        schema = json_schema_to_model(schema, schema.get("$defs", {}))

        results = []

        for image in images:
            image_decoded = base64.b64encode(image.data).decode('utf-8')
            prompt = params["prompt"].format(schema=schema)
            response = model.generate_content([{'mime_type':'image/png', 'data': image_decoded}, prompt], stream=False)
            results.append(
                ExtractorOutput(
                    path=image.path,
                    data=response.text,
                    type="GeminiVisualExtractor",
                    exception="",
                )
            )
        return results
