from sparkpdf.image.DataToImage import DataToImage

def test_data_to_image(spark_session, raw_image_df):
    to_image = DataToImage()
    to_image.setInputCol("content")
    to_image.setOutputCol("image")
    result = to_image.transform(raw_image_df).collect()

    assert (len(result) == 1)
    # present image field
    assert (hasattr(result[0], "image"))
    # image has right origin field
    assert (result[0].image.path == result[0].path)
