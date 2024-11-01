<img src="./images/SparkPdfLogo.png">

<p align="center">
    <a href="https://circleci.com/gh/stabrise/spark-pdf"><img alt="Build" src="https://img.shields.io/circleci/build/github/stabrise/spark-pdf/main"></a>
    <a href="https://github.com/stabrise/spark-pdf/blob/main/LICENSE"><img alt="GitHub" src="https://img.shields.io/github/license/stabrise/spark-pdf.svg?color=blue"></a>
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

## Installation

### Requirements

- Python 3.11
- Apache Spark 3.5 or higher
- Java 8
- Tesseract 5.0 or higher

```bash
  pip install spark-pdf
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
