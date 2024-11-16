<img src="./images/SparkPdfLogo.png">

<p align="center">
    <a href="https://pypi.org/project/pyspark-pdf/" alt="Package on PyPI"><img src="https://img.shields.io/pypi/v/pyspark-pdf.svg" /></a>
    <a href="https://github.com/stabrise/spark-pdf/blob/main/LICENSE"><img alt="GitHub" src="https://img.shields.io/github/license/stabrise/spark-pdf.svg?color=blue"></a>
    <a href="https://stabrise.com"><img alt="StabRise" src="https://img.shields.io/badge/powered%20by-StabRise-orange.svg?style=flat&colorA=E1523D&colorB=007D8A"></a>
</p>



# Spark Pdf

Spark-Pdf is a library for processing documents using Apache Spark.

It includes the following features:

- Load PDF documents/Images
- Extract text from PDF documents/Images
- Extract images from PDF documents
- OCR Images/PDF documents
- Run NER on text extracted from PDF documents/Images
- Visualize NER results

## Support various OCR Engines

|                   | Bbox  level | Support GPU | Separate model  for text detection | Processing time 1 page (CPU/GPU) secs | Support Handwritten Text |
|-------------------|-------------|-------------|------------------------------------|---------------------------------------|--------------------------|
| [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)     | character   | no          | no                                 | 0.2/no                                | not good                 |
| Tesseract OCR CLI | character   | no          | no                                 | 0.2/no                                | not good                 |
| [Easy OCR](https://github.com/JaidedAI/EasyOCR)          | word        | yes         | yes                                |                                       |                          |
| [Surya OCR](https://github.com/VikParuchuri/surya)         | line        | yes         | yes                                |                                       |                          |
| [DocTR](https://github.com/mindee/doctr)       | word        | yes         | yes                                |                                       |                          |

## Installation

### Requirements

- Python 3.10
- Apache Spark 3.5 or higher
- Java 8
- Tesseract 4.0 or higher

```bash
  pip install pyspark-pdf
```

## Development

### Setup

```bash
  git clone
  cd spark-pdf
```

### Install dependencies

```bash
  poetry install
```

### Run tests

```bash
  poetry run pytest --cov=sparkpdf --cov-report=html:coverage_report tests/ 
```

### Build package

```bash
  poetry build
```

### Build documentation

```bash
  poetry run sphinx-build -M html source build
```

### Docker

Build image:

```bash
  docker build -t spark-pdf .
```

Run container:
```bash
  docker run --rm -it --entrypoint bash spark-pdf:latest
```

```bash
docker run -p 8888:8888 spark-pdf:latest

```

### Release

```bash
  poetry version patch
  poetry publish --build
```
