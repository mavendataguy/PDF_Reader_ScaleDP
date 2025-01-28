from scaledp import ScaleDPSession


def test_start_scaledp() -> None:
    """Test the start of a scaledp session."""
    ScaleDPSession(conf={"spark.executor.memory": "2g"}, with_aws=True)
