import streamlit as st
import cv2
from PIL import Image
import numpy as np
import os
import pandas as pd
import json
from paddleocr import PaddleOCR

# Constants
FONT_PATH = 'data/fonts/latin.ttf'
DATA_DIR = os.path.join("data", "ocr")
os.makedirs(DATA_DIR, exist_ok=True)

def preprocess_image(image_np, apply_grayscale=True, apply_thresholding=True, apply_noise_reduction=True, resize_width=None):
    """
    Preprocess the image for OCR.
    """
    if apply_grayscale:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)  # Convert to grayscale

    if apply_noise_reduction:
        image_np = cv2.GaussianBlur(image_np, (5, 5), 0)  # Apply Gaussian blur for noise reduction

    if apply_thresholding:
        _, image_np = cv2.threshold(image_np, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # Apply binary thresholding

    if resize_width:
        height, width = image_np.shape[:2]
        aspect_ratio = width / height
        new_height = int(resize_width / aspect_ratio)
        image_np = cv2.resize(image_np, (resize_width, new_height))  # Resize the image

    return image_np

def draw_bounding_boxes(image_np, boxes):
    """
    Draw bounding boxes on the image.
    """
    image_with_boxes = image_np.copy()
    for box in boxes:
        box = np.array(box, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(image_with_boxes, [box], isClosed=True, color=(0, 255, 0), thickness=2)
    return image_with_boxes

def save_extracted_text(file_path, extracted_data, file_format="csv"):
    """
    Save extracted text to a file in CSV or JSON format.
    """
    if file_format == "csv":
        df = pd.DataFrame(extracted_data)
        df.to_csv(file_path, index=False)
    elif file_format == "json":
        with open(file_path, "w") as f:
            json.dump(extracted_data, f, indent=4)

def save_image_with_boxes(file_path, image_with_boxes):
    """
    Save the image with bounding boxes to a file.
    """
    cv2.imwrite(file_path, cv2.cvtColor(image_with_boxes, cv2.COLOR_RGB2BGR))

def process_image(uploaded_file, ocr, apply_preprocessing, preprocessing_config):
    """
    Process a single image and return extracted data and processed image.
    """
    # Save the uploaded file
    file_path = os.path.join(DATA_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load the image
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    # Apply preprocessing if enabled
    if apply_preprocessing:
        image_np = preprocess_image(image_np, **preprocessing_config)

    # Perform OCR using PaddleOCR
    result = ocr.ocr(image_np, cls=True)

    # Extract text, confidence scores, and bounding boxes from OCR result
    extracted_data = []
    boxes = []
    for line in result:
        for word in line:
            text = word[1][0]  # Extracted text
            confidence = word[1][1]  # Confidence score
            box = word[0]  # Bounding box coordinates
            extracted_data.append({
                "Image": uploaded_file.name,
                "Text": text,
                "Confidence": f"{confidence:.2f}",
                "Bounding Box": box
            })
            boxes.append(box)

    # Draw bounding boxes on the image
    image_with_boxes = draw_bounding_boxes(image_np, boxes) if boxes else None

    return extracted_data, image_with_boxes

def display_uploaded_images(uploaded_files):
    """
    Display uploaded images in a grid layout.
    """
    st.subheader("Uploaded Images Preview")
    num_cols = 3  # Number of images per row
    cols = st.columns(num_cols)  # Create a dynamic number of columns

    for i, uploaded_file in enumerate(uploaded_files):
        col_idx = i % num_cols
        with cols[col_idx]:
            st.image(uploaded_file, caption=f"Uploaded Image: {uploaded_file.name}", use_container_width=True)

def ocr_page():
    st.header("📄 OCR with PaddleOCR")

    # Language selection
    ocr_lang = st.selectbox("Select OCR Language", ["en", "fr", "es", "de", "zh"], help="Select the language for OCR.")

    # Initialize PaddleOCR with the selected language
    ocr = PaddleOCR(use_angle_cls=True, lang=ocr_lang)

    # Upload multiple images
    st.subheader("Upload Images")
    uploaded_files = st.file_uploader("Upload images for OCR", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    # Display uploaded images side by side (preview only)
    if uploaded_files:
        display_uploaded_images(uploaded_files)

    # Preprocessing options
    st.subheader("Preprocessing Options")
    apply_preprocessing = st.checkbox("Apply Preprocessing", help="Enable preprocessing steps before OCR.")
    preprocessing_config = {}
    if apply_preprocessing:
        preprocessing_config = {
            "apply_grayscale": st.checkbox("Convert to Grayscale", value=True, help="Convert the image to grayscale."),
            "apply_thresholding": st.checkbox("Apply Thresholding", value=True, help="Apply binary thresholding to enhance text contrast."),
            "apply_noise_reduction": st.checkbox("Apply Noise Reduction", value=True, help="Apply Gaussian blur to reduce noise."),
            "resize_width": st.number_input("Resize Width (pixels)", min_value=100, max_value=2000, value=800, help="Resize the image to the specified width while maintaining aspect ratio.")
        }

    # Batch processing option
    batch_process = st.checkbox("Enable Batch Processing", help="Process all uploaded images in batch mode.")

    # Export format selection
    export_format = st.radio("Export Extracted Text As", ["CSV", "JSON"], help="Choose the format for exporting extracted text.")

    if uploaded_files:
        if batch_process:
            # Process all images in batch mode
            if st.button("Extract Text from All Images"):
                all_extracted_data = []
                processed_images = []  # Store processed images for display

                for uploaded_file in uploaded_files:
                    with st.spinner(f"Processing {uploaded_file.name}..."):
                        try:
                            extracted_data, image_with_boxes = process_image(uploaded_file, ocr, apply_preprocessing, preprocessing_config)
                            all_extracted_data.extend(extracted_data)
                            if image_with_boxes is not None:
                                processed_images.append((uploaded_file.name, image_with_boxes))
                        except Exception as e:
                            st.error(f"An error occurred during OCR for {uploaded_file.name}: {e}")

                # Display all extracted text in a tabular format
                st.subheader("Extracted Text from All Images")
                if all_extracted_data:
                    df = pd.DataFrame(all_extracted_data)
                    st.dataframe(df)  # Display as an interactive table

                    # Save extracted text to a file
                    file_extension = "csv" if export_format == "CSV" else "json"
                    text_file_path = os.path.join(DATA_DIR, f"all_extracted_text.{file_extension}")
                    save_extracted_text(text_file_path, all_extracted_data, file_extension)

                    # Provide a download button for the extracted text
                    with open(text_file_path, "rb") as f:
                        st.download_button(
                            label=f"Download All Extracted Text as {export_format}",
                            data=f,
                            file_name=f"all_extracted_text.{file_extension}",
                            mime="text/csv" if export_format == "CSV" else "application/json"
                        )
                else:
                    st.warning("No text detected in any of the images.")

                # Display processed images side by side
                if processed_images:
                    st.subheader("Processed Images with Detected Text")
                    num_cols = 3  # Number of images per row
                    cols = st.columns(num_cols)  # Create a dynamic number of columns

                    for i, (image_name, image_with_boxes) in enumerate(processed_images):
                        col_idx = i % num_cols
                        with cols[col_idx]:
                            st.image(image_with_boxes, caption=f"Processed Image: {image_name}", use_container_width=True)

                            # Save and provide a download button for the image with bounding boxes
                            image_file_path = os.path.join(DATA_DIR, f"{image_name}_with_boxes.png")
                            save_image_with_boxes(image_file_path, image_with_boxes)

                            with open(image_file_path, "rb") as f:
                                st.download_button(
                                    label=f"Download {image_name} with Bounding Boxes",
                                    data=f,
                                    file_name=f"{image_name}_with_boxes.png",
                                    mime="image/png"
                                )

        else:
            # Single processing with dropdown
            st.subheader("Single Image Processing")
            selected_file_name = st.selectbox(
                "Select an image to process",
                [file.name for file in uploaded_files],
                help="Select an image from the uploaded files to process individually."
            )

            # Find the selected file object
            selected_file = next(file for file in uploaded_files if file.name == selected_file_name)

            if st.button("Extract Text from Selected Image"):
                with st.spinner(f"Performing OCR on {selected_file.name}..."):
                    try:
                        extracted_data, image_with_boxes = process_image(selected_file, ocr, apply_preprocessing, preprocessing_config)

                        # Display extracted text in a tabular format
                        st.subheader("Extracted Text")
                        if extracted_data:
                            df = pd.DataFrame(extracted_data)
                            st.dataframe(df)  # Display as an interactive table
                        else:
                            st.warning("No text detected in the image.")

                        # Display the image with bounding boxes
                        if image_with_boxes is not None:
                            st.subheader("Image with Detected Text")
                            st.image(image_with_boxes, caption="Image with Detected Text", width=300)

                            # Save and provide a download button for the image with bounding boxes
                            image_file_path = os.path.join(DATA_DIR, f"{selected_file.name}_with_boxes.png")
                            save_image_with_boxes(image_file_path, image_with_boxes)

                            with open(image_file_path, "rb") as f:
                                st.download_button(
                                    label="Download Image with Bounding Boxes",
                                    data=f,
                                    file_name=f"{selected_file.name}_with_boxes.png",
                                    mime="image/png"
                                )

                        # Save extracted text to a file
                        file_extension = "csv" if export_format == "CSV" else "json"
                        text_file_path = os.path.join(DATA_DIR, f"{selected_file.name}_extracted_text.{file_extension}")
                        save_extracted_text(text_file_path, extracted_data, file_extension)

                        # Provide a download button for the extracted text
                        with open(text_file_path, "rb") as f:
                            st.download_button(
                                label=f"Download Extracted Text as {export_format}",
                                data=f,
                                file_name=f"{selected_file.name}_extracted_text.{file_extension}",
                                mime="text/csv" if export_format == "CSV" else "application/json"
                            )

                    except Exception as e:
                        st.error(f"An error occurred during OCR for {selected_file.name}: {e}")