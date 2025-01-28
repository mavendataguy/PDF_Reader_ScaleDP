from pyspark.sql import DataFrame


def test_show_image_pdf(
    raw_image_df: DataFrame,
    image_df: DataFrame,
    pdf_df: DataFrame,
) -> None:
    """Test the show_image and show_pdf methods of the image and PDF dataframes."""
    # Display the raw image dataframe
    assert hasattr(
        raw_image_df,
        "show_image",
    ), "Expected Dataframe to have method 'show_image'"
    raw_image_df.show_image()

    # Display the processed image dataframe
    assert hasattr(
        image_df,
        "show_image",
    ), "Expected Dataframe to have method 'show_image'"
    image_df.show_image()

    # Display the PDF dataframe
    assert hasattr(pdf_df, "show_pdf"), "Expected Dataframe to have method 'show_pdf'"
    pdf_df.show_pdf()


def test_show_text(image_df: DataFrame) -> None:
    """Test the show_text method of the image dataframe."""
    # Display the text dataframe
    assert hasattr(
        image_df,
        "show_text",
    ), "Expected Dataframe to have method 'show_text'"
    image_df.show_text()
