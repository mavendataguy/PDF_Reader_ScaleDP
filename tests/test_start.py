from sparkpdf import start

def test_start_function():
    start(conf={"spark.executor.memory": "2g"}, with_aws=True)
