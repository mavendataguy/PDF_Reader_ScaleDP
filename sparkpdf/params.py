from pyspark.ml.param import Param, Params, TypeConverters

class HasImageType(Params):

    imageType = Param(Params._dummy(), "imageType",
                      "Image type.",
                      typeConverter=TypeConverters.toString)

    def setImageType(self, value):
        """
        Sets the value of :py:attr:`imageType`.
        """
        return self._set(imageType=value)

    def getImageType(self):
        """
        Sets the value of :py:attr:`imageType`.
        """
        return self.getOrDefault(self.imageType)

class HasKeepInputData(Params):

    keepInputData = Param(Params._dummy(), "keepInputData",
                        "Keep input data column in output.",
                        typeConverter=TypeConverters.toBoolean)

    def setKeepInputData(self, value):
        """
        Sets the value of :py:attr:`keepInputData`.
        """
        return self._set(keepInputData=value)

    def getKeepInputData(self):
        """
        Sets the value of :py:attr:`keepInputData`.
        """
        return self.getOrDefault(self.keepInputData)


class HasPathCol(Params):
    """
    Mixin for param pathCol: path column name.
    """
    pathCol = Param(Params._dummy(), "pathCol",
                      "Input column name with path of file.",
                      typeConverter=TypeConverters.toString)

    def setPathCol(self, value):
        """
        Sets the value of :py:attr:`pathCol`.
        """
        return self._set(pathCol=value)

    def getPathCol(self) -> str:
        """
        Gets the value of pathCol or its default value.
        """
        return self.getOrDefault(self.pathCol)

class HasInputCols(Params):
    """
    Mixin for param inputCols: input column names.
    """

    inputCols: "Param[List[str]]" = Param(
        Params._dummy(),
        "inputCols",
        "input column names.",
        typeConverter=TypeConverters.toListString,
    )

    def __init__(self) -> None:
        super(HasInputCols, self).__init__()

    def getInputCols(self):
        """
        Gets the value of inputCols or its default value.
        """
        return self.getOrDefault(self.inputCols)

    def setInputCols(self, value):
        """
        Sets the value of :py:attr:`inputCol`.
        """
        return self._set(inputCols=value)

class HasInputCol(Params):
    """
    Mixin for param inputCol: input column name.
    """

    inputCol: "Param[str]" = Param(
        Params._dummy(),
        "inputCol",
        "input column name.",
        typeConverter=TypeConverters.toString,
    )

    def __init__(self) -> None:
        super(HasInputCol, self).__init__()

    def getInputCol(self) -> str:
        """
        Gets the value of inputCol or its default value.
        """
        return self.getOrDefault(self.inputCol)

    def setInputCol(self, value):
        """
        Sets the value of :py:attr:`inputCol`.
        """
        return self._set(inputCol=value)

class HasOutputCol(Params):
    """
    Mixin for param outputCol: output column name.
    """

    outputCol: "Param[str]" = Param(
        Params._dummy(),
        "outputCol",
        "output column name.",
        typeConverter=TypeConverters.toString,
    )

    def __init__(self) -> None:
        super(HasOutputCol, self).__init__()
        self._setDefault(outputCol=self.uid + "__output")

    def getOutputCol(self) -> str:
        """
        Gets the value of outputCol or its default value.
        """
        return self.getOrDefault(self.outputCol)

    def setOutputCol(self, value):
        """
        Sets the value of :py:attr:`outputCol`.
        """
        return self._set(outputCol=value)


class HasResolution(Params):
    resolution = Param(Params._dummy(), "resolution",
                          "Resolution of image.",
                          typeConverter=TypeConverters.toInt)

    POINTS_PER_INCH = 72

    def setResolution(self, value):
        """
        Sets the value of :py:attr:`resolution`.
        """
        return self._set(resolution=value)

    def getResolution(self):
        """
        Gets the value of :py:attr:`resolution`.
        """
        return self.getOrDefault(self.resolution)

class HasPageCol(Params):
    """
    Mixin for param pageCol: path column name.
    """
    pageCol = Param(Params._dummy(), "pageCol",
                      "Page column name.",
                      typeConverter=TypeConverters.toString)

    def setPageCol(self, value):
        """
        Sets the value of :py:attr:`pageCol`.
        """
        return self._set(pageCol=value)

    def getPageCol(self) -> str:
        """
        Gets the value of pageCol or its default value.
        """
        return self.getOrDefault(self.pageCol)