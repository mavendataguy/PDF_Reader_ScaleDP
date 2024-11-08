Quickstart
============

## Extract text from the scanned PDF file using Apache Spark.

Let's start by creating a simple example that uses the `spark-pdf` library to extract text from a PDF file.

Let's import the necessary modules and create a Spark session:

```python
from sparkpdf import *

spark = SparkPdfSession()
```

Next, we will load a PDF file into a Spark DataFrame:

```python
# Open some pdf from resources or any other pdf file
doc_example = files('resources/pdf/SparkPdf.pdf')
df = spark.read.format("binaryFile").load(doc_example)
```

Now, let's define a Spark ML pipeline to extract text from the PDF file:

```python
pipeline = PipelineModel(stages=[
    PdfDataToImage(),
    TesseractOcr()
])
```

Finally, we will run the pipeline and extract the text from the PDF file:

```python
result = pipeline.transform(df).cache()
result.show()
```

For example, to show text from the first page of the PDF file, you can use the following code:

```python
print(result.select("text.text").collect()[0][0])
```

Or simply:

```python
result.show_text(0)
```

That's it! You have successfully extracted text from a PDF file using the `spark-pdf` library.
After that we are able to process your text data using Spark SQL or other Spark ML libraries.

## Extract text from the images using Apache Spark.

In this example, we will show you how to extract text from images using the `spark-pdf` library.

We already have a Spark session, so let's load an image into a Spark DataFrame:

```python
# Open some image from resources or any other image file
doc_example = files('resources/images/SparkPdfLogo.png')
df = spark.read.format("binaryFile").load(doc_example)
```
Let's define a Spark ML pipeline to extract text from the image:

```python
pipeline = PipelineModel(stages=[
    DataToImage(),
    TesseractOcr()
])
```

Now, let's run the pipeline and extract the text from the image:

```python
result = pipeline.transform(df).cache()
result.show()
```

To show the extracted text from the image, you can use the following code:

```python
result.show_text(0)
```
