Advanced Data Validation System

A modern Python-based system that automates the process of validating payment and personally identifiable information (PII) from scanned documents against reference Excel data. This tool combines Optical Character Recognition (OCR) with Large Language Models (LLMs) to intelligently identify discrepancies and generate detailed validation reports.
Key Features

  Handles both payment information (amounts, account numbers, dates) and PII data (names, phone numbers, ages, etc.)
    Uses Tesseract OCR with OpenCV image preprocessing for accurate text extraction
    LLM-Powered Validation: Leverages OpenAI's GPT-4 for context-aware data validation and discrepancy explanation

Configurable validation rules for different data types with automatic action recommendations 
DGenerates comprehensive validation reports with statistics and actionable insights and Includes fallback mechanisms for when external services are unavailable

How It Works

  Document Processing: Scanned documents are enhanced and processed using OpenCV for OCR accuracy and Tesseract OCR extracts text from the preprocessed images
  An LLM parses the extracted text into structured data based on the specified schema (payment or PII) then Extracted data is compared against reference Excel data using type-aware validation rules
    Reporting: The system generates detailed reports highlighting discrepancies with recommended actions (auto-correct, manual review, or valid)

Prerequisites
Python 3.8+
Tesseract OCR
OpenAI API key

Steps

Clone the repository:

bash

git clone https://github.com/yourusername/data-validation-system.git
cd data-validation-system

  Install required Python packages:

bash

pip install -r requirements.txt

  Install Tesseract OCR:

  Windows: Download from UB-Mannheim/tesseract

  macOS: brew install tesseract

  Linux: sudo apt-get install tesseract-ocr

    Set your OpenAI API key as an environment variable:

export OPENAI_API_KEY='your-api-key-here'
