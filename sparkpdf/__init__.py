import os
import sys
import pyspark
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.ml.pipeline import PipelineModel
from importlib.resources import files
from sparkpdf.image.DataToImage import DataToImage
from sparkpdf.pdf.PdfDataToImage import PdfDataToImage
from sparkpdf.models.recognizers.TesseractOcr import TesseractOcr
from sparkpdf.models.ner.Ner import Ner
from sparkpdf.image.ImageDrawBoxes import ImageDrawBoxes
from sparkpdf.text.TextToDocument import TextToDocument

from sparkpdf import enums
from sparkpdf.enums import *

from pyspark.sql import DataFrame


from sparkpdf.utils.show_utils import show_image, show_pdf, show_ner, visualize_ner, show_text

DataFrame.show_image = lambda self, column="", limit=5, width=None, show_meta=True, : show_image(self, column, limit, width, show_meta)
DataFrame.show_pdf = lambda self, column="", limit=5, width=None, show_meta=True, : show_pdf(self, column, limit, width, show_meta)
DataFrame.show_ner = lambda self, column="ner", limit=20, truncate=True: show_ner(self, column, limit, truncate)
DataFrame.show_text = lambda self, column="", limit=20, width=None: show_text(self, column, limit, width)
DataFrame.visualize_ner = lambda self, column="ner", text_column="text", limit=20, width=None, labels_list=None : visualize_ner(self, column, text_column, limit, width, labels_list)

__all__ = ['SparkPdfSession',
           'DataToImage',
           'ImageDrawBoxes',
           'PdfDataToImage',
           'TesseractOcr',
           'Ner',
           'TextToDocument',
           'PipelineModel',
           ] + dir(enums)


def version():
    version_path = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(version_path, 'VERSION'), encoding="utf-8") as version_file:
        return version_file.read().strip()


__version__ = version()


def aws_version():
    spark_hadoop_map = {"3.0": "2.7.4",
                        "3.1": "3.2.0",
                        "3.2": "3.3.1",
                        "3.3": "3.3.2",
                        "3.4": "3.3.4",
                        "3.5": "3.3.4"}
    return spark_hadoop_map[pyspark.__version__[:3]]


def SparkPdfSession(conf=None,
                    master_url="local[*]",
                    with_aws=False,
                    with_pro=False,
                    logLevel="ERROR"):
    """
    Start Spark session with SparkPDF
    @param conf: Instance of SparkConf or dict with extra configuration.
    @param master_url: Spark master URL
    @param with_aws: Enable AWS support
    @param logLevel: Log level
    """
    os.environ['PYSPARK_PYTHON'] = sys.executable
    os.environ["TRANSFORMERS_VERBOSITY"] = logLevel.lower()

    if with_pro:
        try:
            import sparkpdf_pro
        except ImportError:
            raise ImportError("Spark Pdf Pro is not installed. Please install it using 'pip install sparkpdf-pro'")


    jars = []
    jars_packages = []
    default_conf = {
        "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
        "spark.kryoserializer.buffer.max": "200M",
        "spark.driver.memory": "8G"
    }

    if with_aws:
        jars_packages.append("org.apache.hadoop:hadoop-aws:" + aws_version())

    if conf:
        if not isinstance(conf, dict):
            conf = dict(conf.getAll())
        default_conf.update(conf)
        extra_jars_packages = default_conf.get("spark.jars.packages")
        if extra_jars_packages:
            jars_packages.append(extra_jars_packages)
        extra_jars = default_conf.get("spark.jars")
        if extra_jars:
            jars.append(extra_jars)

    builder = SparkSession.builder \
        .master(master_url) \
        .appName("SparkPdf")

    for k, v in default_conf.items():
        builder.config(str(k), str(v))

    builder.config("spark.jars", ",".join(jars))
    builder.config("spark.jars.packages", ",".join(jars_packages))

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel(logLevel=logLevel)
    return spark
