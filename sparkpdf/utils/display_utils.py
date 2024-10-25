
from pyspark.sql import DataFrame
from sparkpdf import DataToImage, PdfDataToImage
import pyspark.sql.functions as f
import random

def _show_image(image, width=600, show_meta=True, index=0):
    from IPython.display import display
    from sparkpdf.schemas.Image import Image
    if image is None:
        print("Empty image")
        return
    if show_meta:
        print(f"""
    Image#: {index}
    Path: {image.path.split("/")[-1]}
    Resolution: {image.resolution} dpi
    Size: {image.width} x {image.height} px""")
    image = Image(**image.asDict())
    factor = width / image.width
    display(image.to_pil().resize(size=(width, int(image.height * factor))), metadata={"width": width})

def get_column_type(df: DataFrame, column_name: str) -> str:
    for name, dtype in df.dtypes:
        if name == column_name:
            return dtype
    return None


def show_image(df, field="image", limit=5,  width=600, show_meta=True):
    column_type = get_column_type(df, field)
    if column_type == "binary":
        df = DataToImage().setInputCol(field).setOutputCol("image").transform(df)
        field = "image"
    for id_, row in enumerate(df.limit(limit).select(field).collect()):
        image = row[field]
        _show_image(image, width, show_meta, id_)


def show_pdf(df, field="content", limit=5,  width=600, show_meta=True):
    column_type = get_column_type(df, field)
    if column_type == "binary":
        df = PdfDataToImage(inputCol=field).transform(df)
        field = "image"
    else:
        raise ValueError("Field must be binary")
    for id_, row in enumerate(df.limit(limit).select(field).collect()):
        image = row[field]
        _show_image(image, width, show_meta, id_)


def show_ner(df, column="ner", limit=20, truncate=False):
    df.select(f.explode(f"{column}.entities").alias("ner")).select("ner.*").show(limit, truncate=truncate)


def visualize_ner(df, column="ner", text_column="text", limit=20, labels_list=None):
    from IPython.display import display, HTML
    STYLE_CONFIG = f"""
<style>
    .spark-pdf-display-entity-wrapper{{
        display: inline-grid;
        text-align: center;
        border-radius: 8px;
        margin: 0 2px 5px 2px;
        padding: 1px
    }}

    .spark-pdf-display-text{{
        font-size: 14px;
        line-height: 18px;
        font-family: sans-serif !important;
        background: #f1f2f3;
        border-width: medium;
        text-align: center;
        font-weight: 400;
        border-radius: 5px;
        padding: 2px 5px;
        display: block;
        margin: 2px;
    }}

    .spark-pdf-display-entity-name{{
        font-size: 10px;
        line-height: 16px;
        color: #ffffff;
        font-family: sans-serif !important;
        text-transform: uppercase;
        font-weight: 500;
        display: block;
        padding: 1px 4px;
    }}
    
    .spark-pdf-display-others{{
        font-size: 14px;
        line-height: 24px;
        font-family: sans-serif !important;
        font-weight: 400;
    }}
</style>
"""
    df = df.limit(limit).select(column, text_column).cache()
    entities = df.select(f.explode(f"{column}.entities").alias("ner")).select("ner.*").collect()
    original_text = df.select(text_column).collect()[0][0].text
    if labels_list is not None:
        labels_list = [v.lower() for v in labels_list]
    label_color = {}
    html_output = ""
    pos = 0
    df.unpersist()

    def get_color(entity_group):
        return "#{:06x}".format(random.randint(0, 0xFFFFFF))

    for entity in entities:
        entity_group = entity.entity_group
        if (entity_group not in label_color) and ((labels_list is None) or (entity_group.lower() in labels_list)):
            label_color[entity_group.lower()] = get_color(entity_group)

        begin = int(entity.start)
        end = int(entity.end)
        if pos < begin and pos < len(original_text):
            white_text = original_text[pos:begin]
            html_output += '<span class="spark-pdf-display-others" style="background-color: white">{}</span>'.format(
                white_text)
        pos = end + 1
        if entity_group.lower() in label_color:
            html_output += '<span class="spark-pdf-display-entity-wrapper" style="background-color: {}"><span class="spark-pdf-display-text">{} </span><span class="spark-pdf-display-entity-name">{}</span></span>'.format(
                label_color[entity_group.lower()],
                original_text[begin:end + 1],
                entity_group)
        else:
            html_output += '<span class="spark-pdf-display-others" style="background-color: white">{}</span>'.format(
                original_text[begin:end + 1])

    if pos < len(original_text):
        html_output += '<span class="spark-pdf-display-others" style="background-color: white">{}</span>'.format(
            original_text[pos:])

    html_output += """</div>"""

    html_output = STYLE_CONFIG + html_output.replace("\n", "<br>")

    display(HTML(html_output))
