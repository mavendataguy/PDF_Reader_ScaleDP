from scaledp import ScaleDPSession

def test_start_scaledp():
    ScaleDPSession(conf={"spark.executor.memory": "2g"}, with_aws=True)
