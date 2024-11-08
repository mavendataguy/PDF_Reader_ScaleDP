from sparkpdf import DataToImage


def test_local_pipeline(image_file):
    from sparkpdf.pipeline.PandasPipeline import PandasPipeline, UserDefinedFunction
    import pyspark
    original_udf = pyspark.sql.udf.UserDefinedFunction

    try:
        # Replace UserDefinedFunction with the custom one
        pyspark.sql.udf.UserDefinedFunction = UserDefinedFunction

        # Initialize the DataToImage stage
        data_to_image = DataToImage()

        # Initialize the LocalPipeline with the DataToImage stage
        pipeline = PandasPipeline(stages=[data_to_image])

        # Read the image file
        with open(image_file, "rb") as f:
            image_data = f.read()

        # Transform the image data through the pipeline
        result = pipeline.fromBinary(image_data)

        # Verify the pipeline result
        assert result is not None, "Expected a non-None result from the pipeline"
        assert len(result) > 0, "Expected at least one result from the pipeline"

    finally:
        # Restore the original UserDefinedFunction
        pyspark.sql.udf.UserDefinedFunction = original_udf
