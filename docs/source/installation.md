Installation
============

To install the `ScaleDP` library, follow the steps below:

## Prerequisites

- Python 3.11 or higher
- PySpark 3.0 or higher
- Tesseract 5.0 or higher

## Installation using pip

You can install the `ScaleDP` library using `pip`. Run the following command in your terminal:

```sh
pip install pyspark-pdf
```

For run ML transformers you need to install extra dependencies using following command:

```sh
pip install pyspark-pdf[ml]
```

Or manually install `transformers` and `torch` library:

```sh
pip install transformers torch
```

## Installation using docker

You can also use the `Dockerfile` provided in the repository to build a Docker image. To build the image, run the following command:

```sh
docker build -t scaledp .
```

## Installation from source

To install the `ScaleDP` library from source, clone the repository and run the following command:

```sh
git clone https://github.com/StabRise/scaledp.git
cd scaledp
pip install .
```
