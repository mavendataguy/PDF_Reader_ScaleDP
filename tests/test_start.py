from sparkpdf import sparkpdf

def test_start_function():
    sparkpdf(conf={"spark.executor.memory": "2g"}, with_aws=True)
