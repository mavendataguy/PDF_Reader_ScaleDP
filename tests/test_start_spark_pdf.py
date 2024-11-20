from scaledp import SparkPdfSession

def test_start_sparkpdf():
    SparkPdfSession(conf={"spark.executor.memory": "2g"}, with_aws=True)
