

def test_show_image_pdf(raw_image_df, image_df, pdf_df):
    # Display the raw image dataframe
    assert hasattr(raw_image_df, "show_image"), "Expected Dataframe to have method 'show_image'"
    raw_image_df.show_image()

    # Display the processed image dataframe
    assert hasattr(image_df, "show_image"), "Expected Dataframe to have method 'show_image'"
    image_df.show_image()

    # Display the PDF dataframe
    assert hasattr(pdf_df, "show_pdf"), "Expected Dataframe to have method 'show_pdf'"
    pdf_df.show_pdf()
