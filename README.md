
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
</p>

---

**Tutorials**: <a href="https://github.com/StabRise/ScaleDP-Tutorials/" target="_blank">https://github.com/StabRise/ScaleDP-Tutorials</a>

**Source Code**: <a href="https://github.com/StabRise/ScaleDP/" target="_blank">https://github.com/StabRise/ScaleDP</a>

---

# Welcome to the ScaleDP library

ScaleDP is library allows you to process documents using Apache Spark.  Discover pre-trained models for your projects or play with the thousands of machine learning apps hosted on the [Hugging Face Hub](https://huggingface.co/).

## Key features:

- Load PDF documents/Images to the Spark DataFrame
- Extract text from PDF documents/Images
- Extract images from PDF documents
- OCR Images/PDF documents using various OCR engines
- Run NLP models from the Hugging Face Hub on text extracted from PDF documents/Images
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

Install the `ScaleDP` package with [pip](https://pypi.org/project/pyspark-pdf/):

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

Read example pdf file:

```python
pdf_example = files('resources/pdfs/example.pdf')

df = spark.read.format("binaryFile") \
    .load(pdf_example)

df.show_pdf()
```
Output:

<img src="images/PdfOutput.png" width="400">

Define pipeline for extract text from the PDF and call it:

```python
pipeline = PipelineModel(stages=[
    PdfDataToImage(),
    TesseractOcr(keepFormatting=True, psm=PSM.SPARSE_TEXT)
])

result = pipeline.transform(df).cache()
result.show_text()
```

Output:

<img src="images/TextOutput.png" width="400">

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
