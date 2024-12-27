
<p align="center">
  <br/>
    <img alt="ScaleDP" src="https://stabrise.com/media/filer_public_thumbnails/filer_public/4a/7d/4a7d97c2-50d7-4b7a-9902-af2df9b574da/scaledplogo.png__1000x300_subsampling-2.webp" width="450" style="max-width: 100%;">
  <br/>
</p>

<p align="center">
    <i>An Open-Source Library for Processing Documents in Apache Spark.</i>
</p>

<p align="center">
    <a href="https://pypi.org/project/scaledp/" alt="Package on PyPI"><img src="https://img.shields.io/pypi/v/scaledp.svg" /></a>
    <a href="https://github.com/stabrise/spark-pdf/blob/main/LICENSE"><img alt="GitHub" src="https://img.shields.io/github/license/stabrise/spark-pdf.svg?color=blue"></a>
    <a href="https://stabrise.com"><img alt="StabRise" src="https://img.shields.io/badge/powered%20by-StabRise-orange.svg?style=flat&colorA=E1523D&colorB=007D8A"></a>
    <a href="https://app.codacy.com/gh/StabRise/ScaleDP/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade">
    <img src="https://app.codacy.com/project/badge/Grade/98570508281140c2a33e616a4f749c20" alt="Codacy Badge" />
</a></p>

---

**Source Code**: <a href="https://github.com/StabRise/ScaleDP/" target="_blank">https://github.com/StabRise/ScaleDP</a>

**Quickstart**: <a href="https://colab.research.google.com/github/StabRise/scaledp-tutorials/blob/master/1.QuickStart.ipynb" target="_blank">1.QuickStart.ipynb</a>

**Tutorials**: <a href="https://github.com/StabRise/ScaleDP-Tutorials/" target="_blank">https://github.com/StabRise/ScaleDP-Tutorials</a>

---

# Welcome to the ScaleDP library

ScaleDP is library allows you to process documents using Apache Spark.  Discover pre-trained models for your projects or play with the thousands of machine learning apps hosted on the [Hugging Face Hub](https://huggingface.co/).

## Key features

### Document processing:
- Load PDF documents/Images to the Spark DataFrame
- Extract text from PDF documents/Images
- Extract images from PDF documents
- Create document processing pipelines

### OCR:
- OCR Images/PDF documents using various OCR engines
- OCR Images/PDF documents using Vision LLM models

### CV:
- Object detection on images
- Text detection on images

### NLP and LLM:
- Extract data from the image using Vision LLM models
- Extract data from the text/images using LLM models
- Extract data from using DSPy framework
- Extract data from the text/images using NLP models from the Hugging Face Hub
- Visualize results

Support various open-source OCR engines:

 - [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) 
 - [Easy OCR](https://github.com/JaidedAI/EasyOCR)   
 - [Surya OCR](https://github.com/VikParuchuri/surya) 
 - [DocTR](https://github.com/mindee/doctr) 

## Installation

### Prerequisites

- Python 3.10 or higher
- Apache Spark 3.5 or higher
- Java 8
- Tesseract 4.0 or higher

### Installation using pip

Install the `ScaleDP` package with [pip](https://pypi.org/project/scaledp/):

```bash
pip install scaledp
```

### Installation using Docker

Build image:

```bash
  docker build -t scaledp .
```

Run container:
```bash
  docker run -p 8888:8888 scaledp:latest
```

Open Jupyter Notebook in your browser:
```bash
  http://localhost:8888
```

## Qiuckstart

Start a Spark session with ScaleDP:

```python
from scaledp import *
spark = ScaleDPSession()
spark
```

Read example image file:

```python
image_example = files('resources/images/Invoice.png')
df = spark.read.format("binaryFile") \
    .load(image_example)

df.show_image("content")
```
Output:

<img src="https://github.com/StabRise/ScaleDP/blob/master/images/ImageOutput.png?raw=true" width="400">

Define pipeline for extract text from the image and run NER:

```python
pipeline = PipelineModel(stages=[
    DataToImage(inputCol="content", outputCol="image"),
    TesseractOcr(inputCol="image", outputCol="text", psm=PSM.AUTO, keepInputData=True),
    Ner(model="obi/deid_bert_i2b2", inputCol="text", outputCol="ner", keepInputData=True),
    ImageDrawBoxes(inputCols=["image", "ner"], outputCol="image_with_boxes", lineWidth=3, 
                   padding=5, displayDataList=['entity_group'])
])

result = pipeline.transform(df).cache()

result.show_text("text")
```

Output:

<img src="https://github.com/StabRise/ScaleDP/blob/master/images/TextOutput.png?raw=true" width="400">

Show NER results:

```python
result.show_ner(limit=20)
```

Output:
```text
+------------+-------------------+----------+-----+---+--------------------+
|entity_group|              score|      word|start|end|               boxes|
+------------+-------------------+----------+-----+---+--------------------+
|        HOSP|  0.991257905960083|  Hospital|    0|  8|[{Hospital:, 0.94...|
|         LOC|  0.999171257019043|    Dutton|   10| 16|[{Dutton,, 0.9609...|
|         LOC| 0.9992585778236389|        MI|   18| 20|[{MI, 0.93335297,...|
|          ID| 0.6838774085044861|        26|   29| 31|[{26-123123, 0.90...|
|       PHONE| 0.4669836759567261|         -|   31| 32|[{26-123123, 0.90...|
|       PHONE| 0.7790696024894714|    123123|   32| 38|[{26-123123, 0.90...|
|        HOSP|0.37445762753486633|      HOPE|   39| 43|[{HOPE, 0.9525460...|
|        HOSP| 0.9503226280212402|     HAVEN|   44| 49|[{HAVEN, 0.952546...|
|         LOC| 0.9975488185882568|855 Howard|   59| 69|[{855, 0.94682700...|
|         LOC| 0.9984399676322937|    Street|   70| 76|[{Street, 0.95823...|
|        HOSP| 0.3670221269130707|  HOSPITAL|   77| 85|[{HOSPITAL, 0.959...|
|         LOC| 0.9990363121032715|    Dutton|   86| 92|[{Dutton,, 0.9647...|
|         LOC|  0.999313473701477|  MI 49316|   94|102|[{MI, 0.94589012,...|
|       PHONE| 0.9830010533332825|   ( 123 )|  110|115|[{(123), 0.595334...|
|       PHONE| 0.9080978035926819|       456|  116|119|[{456-1238, 0.955...|
|       PHONE| 0.9378324151039124|         -|  119|120|[{456-1238, 0.955...|
|       PHONE| 0.8746233582496643|      1238|  120|124|[{456-1238, 0.955...|
|     PATIENT|0.45354968309402466|hopedutton|  132|142|[{hopedutton@hope...|
|       EMAIL|0.17805588245391846| hopehaven|  143|152|[{hopedutton@hope...|
|        HOSP|  0.505658745765686|   INVOICE|  157|164|[{INVOICE, 0.9661...|
+------------+-------------------+----------+-----+---+--------------------+
```

Visualize NER results:

```python
result.visualize_ner(labels_list=["DATE", "LOC"])
```
<img src="https://github.com/StabRise/ScaleDP/blob/master/images/NerVisual.png?raw=true" width="400">

Original image with NER results:

```python
result.show_image("image_with_boxes")
```
<img src="https://github.com/StabRise/ScaleDP/blob/master/images/NerVisualOnImage.png?raw=true" width="400">

## Ocr engines

|                   | Bbox  level | Support GPU | Separate model  for text detection | Processing time 1 page (CPU/GPU) secs | Support Handwritten Text |
|-------------------|-------------|-------------|------------------------------------|---------------------------------------|--------------------------|
| [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)     | character   | no          | no                                 | 0.2/no                                | not good                 |
| Tesseract OCR CLI | character   | no          | no                                 | 0.2/no                                | not good                 |
| [Easy OCR](https://github.com/JaidedAI/EasyOCR)          | word        | yes         | yes                                |                                       |                          |
| [Surya OCR](https://github.com/VikParuchuri/surya)         | line        | yes         | yes                                |                                       |                          |
| [DocTR](https://github.com/mindee/doctr)       | word        | yes         | yes                                |                                       |                          |


## Disclaimer

This project is not affiliated with, endorsed by, or connected to the Apache Software Foundation or Apache Spark.
