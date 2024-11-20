.. scaledp documentation master file

ScaleDP documentation
=======================

ScaleDP is a library for processing documents using Apache Spark.

It includes the following features:

- Load PDF documents/Images
- Extract images from PDF documents
- Extract text from PDF documents/Images
- Run NLP models on text extracted from PDF documents/Images
- Visualize NLP results on the images

Benefits of using ScaleDP:

- Scalable: ScaleDP is built on top of Apache Spark, which is a distributed computing framework. This allows you to process large volumes of documents in parallel.
- Fast: ScaleDP is built for speed. It uses Spark's distributed computing capabilities to process documents quickly.
- Easy to use models from the Hugging Face's Hub: ScaleDP integrates with Hugging Face's Transformers library, which provides state-of-the-art NLP models for text processing.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation.md
   quickstart.md

