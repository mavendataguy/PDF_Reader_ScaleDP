from typing import Any

import itertools


class UserDefinedFunction:
    """
    User defined function in Python

    .. versionadded:: 1.3

    Notes
    -----
    The constructor of this class is not supposed to be directly called.
    Use :meth:`pyspark.sql.functions.udf` or :meth:`pyspark.sql.functions.pandas_udf`
    to create this instance.
    """

    def __init__(
        self,
        func,
        returnType,
        name,
        evalType,
        deterministic: bool = True,
    ):
        if not callable(func):
            raise
        self.func = func
        self.returnType = returnType

    def __call__(self, *cols):
        cols = zip(*cols)
        return [self.func(*i) for i in cols]

    def _wrapped(self):
        return self


def lit(value):
    return itertools.repeat(value)


def _invoke_function(name: str, *args: Any):
    if name == "lit":
        return lit(*args)
    else:
        raise ValueError("Invalid function name: %s" % name)


temp_functions = {}


def pathSparkFunctions(pyspark):
    temp_functions["udf"] = pyspark.sql.udf.UserDefinedFunction
    pyspark.sql.udf.UserDefinedFunction = UserDefinedFunction
    temp_functions["invoke_function"] = pyspark.sql.functions._invoke_function
    pyspark.sql.functions._invoke_function = _invoke_function


def unpathSparkFunctions(pyspark):
    pyspark.sql.udf.UserDefinedFunction = temp_functions["udf"]
    pyspark.sql.functions._invoke_function = temp_functions["invoke_function"]


import pandas as pd


class DatasetPd(pd.DataFrame):

    def withColumn(self, name, col):
        self.insert(0, name, col, True)
        return self

    def drop(self, col):
        return self

    def repartition(self, numPartitions):
        return self

    def coalesce(self, numPartitions):
        return self


class PandasPipeline:
    stages = []

    def setStages(self, value):
        self.stages = value
        return self

    def __init__(self, stages):
        self.setStages(stages)

    def fromFile(self, filename):
        with open(filename, "rb") as f:
            data = f.read()

        # input = Dataset(data = [dict(content=data, path=filename)])

        input = DatasetPd(dict(content=[data], path=[filename], resolution=[0]))

        data = input
        for stage in self.stages:
            data = stage._transform(data)

        return data

    def fromBinary(self, data, filename="memory"):
        input = DatasetPd(dict(content=[data], path=[filename], resolution=[0]))
        data = input
        for stage in self.stages:
            data = stage._transform(data)

        return data
