import os
import cv2
from PIL import Image
import streamlit as st
from paddleocr import PaddleOCR, draw_ocr

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Font path for drawing OCR results
font_path = 'fonts/latin.ttf'

# Streamlit app
st.title("Receipt OCR App")

# File uploader allows user to add their own image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the file to an image
    image = Image.open(uploaded_file).convert('RGB')
    img_path = "uploaded_image.jpg"
    image.save(img_path)
    
    # Perform OCR on the image
    result = ocr.ocr(img_path, cls=True)
    
    # Draw result
    result = result[0]
    boxes = [line[0] for line in result]
    txts = [line[1][0] for line in result]
    scores = [line[1][1] for line in result]
    
    im_show = draw_ocr(image, boxes, txts, scores, font_path=font_path)
    im_show = Image.fromarray(im_show)
    
    # Display the original image
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    # Display the OCR results
    st.subheader("OCR Results")
    for txt, score in zip(txts, scores):
        st.write(f"Text: {txt}, Confidence: {score}")
    
    # Display the OCR annotated image
    st.image(im_show, caption='OCR Annotated Image.', use_column_width=True)

# Note: Ensure you have the necessary dependencies installed and the font file available at the specified path.