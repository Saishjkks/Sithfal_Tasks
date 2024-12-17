import pdfplumber
import pytesseract
from PIL import Image
import pandas as pd
import re
import os

# Path to Tesseract-OCR executable
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def extract_text_with_ocr(pdf_path, page_number):
    """Extracts text from an image-based PDF page using OCR."""
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_number]
        image = page.to_image()
        temp_image_path = "temp_page.png"
        image.save(temp_image_path, format="PNG")

        # Check if the image file exists
        if os.path.exists(temp_image_path):  # Corrected the typo here
            print(f"Image saved as {temp_image_path}. Proceeding with OCR.")
            try:
                text = pytesseract.image_to_string(Image.open(temp_image_path))
                print("OCR Text extracted:")
                print(text)  # Print OCR text for debugging
            finally:
                os.remove(temp_image_path)  # Clean up the temporary file
        else:
            print(f"Failed to save image as {temp_image_path}.")
            text = ""  # Return empty string if image is not saved

    return text

def extract_text(pdf_path, page_number):
    """Extracts text from a specific PDF page, falling back to OCR if necessary."""
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_number]
        text = page.extract_text()
        if not text:  # If no text, try OCR
            print(f"No text found on Page {page_number + 1}, performing OCR...")
            text = extract_text_with_ocr(pdf_path, page_number)
    return text

def preprocess_text(text):
    """Cleans and normalizes the extracted text."""
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces and line breaks
    text = re.sub(r'[^\x00-\x7F]+', '', text)  # Remove non-ASCII characters
    return text.strip()

def parse_unemployment_info(text):
    """Parses unemployment information by degree from extracted text."""
    pattern = r"(Professional degree|Master's degree|Bachelor's degree|Associate's degree|High school diploma):?\s*(\d+)"
    matches = re.findall(pattern, text, re.IGNORECASE)
    info = {degree.strip(): count.strip() for degree, count in matches}
    return info

def extract_table(pdf_path, page_number):
    """Extracts and formats tables from a specific PDF page."""
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_number]
        tables = page.extract_tables()
        if tables:
            # Convert the first table to a DataFrame
            table = tables[0]
            df = pd.DataFrame(table[1:], columns=table[0])  # Use first row as column headers
            return df
        else:
            return None

def format_output(degree_info, table_data):
    """Formats the output for both tasks in the desired manner."""
    # Page 2: Unemployment Information
    print("Page 2: Unemployment Information by Degree")
    print("-----------------------------------------")
    if degree_info:
        for degree, count in degree_info.items():
            print(f"{degree}: {count}")
    else:
        print("No unemployment information found.")

    # Page 6: Tabular Data
    print("\nPage 6: Tabular Data")
    print("--------------------")
    if table_data is not None:
        print("Extracted Tabular Data:")
        print(table_data.to_string(index=False))
    else:
        print("No table found on Page 6.")

def main(pdf_path):
    # Task 1: Extract unemployment info from Page 2
    print("Extracting data from Page 2...")
    page_2_text = extract_text(pdf_path, 1)  # Adjusted to index 1 (Page 2)
    cleaned_text = preprocess_text(page_2_text)
    degree_info = parse_unemployment_info(cleaned_text)

    # Task 2: Extract tabular data from Page 6
    print("\nExtracting tabular data from Page 6...")
    table_data = extract_table(pdf_path, 5)  # Adjusted to index 5 (Page 6)

    # Format and display the output
    format_output(degree_info, table_data)

# Path to your PDF file
pdf_path = r"C:\Users\SAISH\Desktop\sitha\sample.pdf"

main(pdf_path)
