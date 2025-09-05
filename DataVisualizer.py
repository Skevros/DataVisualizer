import pytesseract
from PIL import Image
import cv2
import numpy as np
import pandas as pd
import openai
import json
import re
import os
from datetime import datetime

# Initialize OpenAI client
# REMINDER: Always use environment variables for API keys
# Consider adding error handling for missing API keys
try:
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    if not client.api_key:
        raise ValueError("OpenAI API key not found in environment variables")
except Exception as e:
    print(f"Error initializing OpenAI client: {e}")
    client = None


class AdvancedDataValidator:
    """Validates payment and PII data from documents against reference Excel data."""
    
    def __init__(self):
        self.reference_data = None
        # REMINDER: Consider making validation rules configurable via external file
        self.validation_rules = {
            "payment": {
                "amount": {"type": "numeric", "auto_correct": True},
                "account_number": {"type": "alphanumeric", "auto_correct": False},
                "routing_number": {"type": "numeric", "auto_correct": False},
                "payment_date": {"type": "date", "auto_correct": True}
            },
            "pii": {
                "name": {"type": "text", "auto_correct": True},
                "date_seen": {"type": "date", "auto_correct": True},
                "phone_number": {"type": "phone", "auto_correct": True},
                "age": {"type": "numeric", "auto_correct": True},
                "gender": {"type": "category", "values": ["male", "female", "other"], "auto_correct": True},
                "pediatric": {"type": "boolean", "auto_correct": False}
            }
        }
    
    def load_reference_data(self, excel_path):
        """Load reference data from Excel file.
        
        Args:
            excel_path (str): Path to Excel file
            
        Returns:
            pandas.DataFrame: Loaded reference data
            
        Raises:
            FileNotFoundError: If Excel file doesn't exist
            ValueError: If Excel file is invalid
        """
        # REMINDER: Add support for other file formats (CSV, Google Sheets)
        if not os.path.exists(excel_path):
            raise FileNotFoundError(f"Reference file not found: {excel_path}")
            
        try:
            self.reference_data = pd.read_excel(excel_path)
            return self.reference_data
        except Exception as e:
            raise ValueError(f"Error reading Excel file: {e}")
    
    def preprocess_image(self, image_path):
        """Enhance image quality for better OCR results.
        
        Args:
            image_path (str): Path to image file
            
        Returns:
            numpy.ndarray: Processed image
            
        Raises:
            FileNotFoundError: If image file doesn't exist
        """
        # REMINDER: Add PDF support using PyMuPDF in future versions
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not read image: {image_path}")
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply adaptive thresholding
            processed = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Noise removal
            kernel = np.ones((1, 1), np.uint8)
            processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel)
            processed = cv2.medianBlur(processed, 3)
            
            return processed
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            # Fallback: return original image if processing fails
            return cv2.imread(image_path, 0)  # Read as grayscale
    
    def extract_text_with_ocr(self, image_path):
        """Extract text from image using Tesseract OCR.
        
        Args:
            image_path (str): Path to image file
            
        Returns:
            str: Extracted text
        """
        try:
            processed_image = self.preprocess_image(image_path)
            
            # Custom configuration for better results
            custom_config = r'--oem 3 --psm 6'
            text = pytesseract.image_to_string(processed_image, config=custom_config)
            
            return text
        except Exception as e:
            print(f"Error in OCR extraction: {e}")
            return ""
    
    def extract_structured_data(self, extracted_text, data_type):
        """Use LLM to extract structured data from OCR text.
        
        Args:
            extracted_text (str): Text extracted from document
            data_type (str): Type of data ('payment' or 'pii')
            
        Returns:
            dict: Structured data extracted from text
        """
        # REMINDER: Consider adding a fallback method if LLM is unavailable
        if client is None:
            print("OpenAI client not available, using rule-based extraction")
            return self._rule_based_extraction(extracted_text, data_type)
        
        if data_type == "payment":
            prompt = (
                "Extract payment information from the text. Return JSON with fields: "
                "amount, account_number, routing_number, payment_date, payer_name. "
                "Use null for missing values. Text: " + extracted_text
            )
        elif data_type == "pii":
            prompt = (
                "Extract PII information from the text. Return JSON with fields: "
                "name, date_seen, phone_number, age, gender, pediatric. "
                "Format dates as YYYY-MM-DD. For pediatric, use true/false. "
                "Use null for missing values. Text: " + extracted_text
            )
        else:
            raise ValueError("Unknown data type. Use 'payment' or 'pii'")
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You extract structured information from text accurately."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            return result
        except Exception as e:
            print(f"Error in LLM extraction: {e}")
            return self._rule_based_extraction(extracted_text, data_type)
    
    def _rule_based_extraction(self, text, data_type):
        """Fallback method for data extraction without LLM.
        
        Args:
            text (str): Text to extract data from
            data_type (str): Type of data to extract
            
        Returns:
            dict: Extracted data
        """
        # REMINDER: Expand this fallback method for better coverage
        result = {}
        
        if data_type == "payment":
            # Simple rule-based extraction for payment data
            result = {
                "amount": self._extract_amount(text),
                "account_number": self._extract_account_number(text),
                "payment_date": self._extract_date(text),
                # Add more fields as needed
            }
        elif data_type == "pii":
            # Simple rule-based extraction for PII data
            result = {
                "name": self._extract_name(text),
                "phone_number": self._extract_phone(text),
                "age": self._extract_age(text),
                # Add more fields as needed
            }
        
        return result
    
    def _extract_amount(self, text):
        """Extract payment amount from text."""
        # Simple amount extraction - expand as needed
        matches = re.findall(r'\$\s*(\d+\.\d{2})', text)
        return matches[0] if matches else None
    
    def _extract_account_number(self, text):
        """Extract account number from text."""
        # Simple account number extraction - expand as needed
        matches = re.findall(r'account\s*[#:]*\s*(\w+)', text, re.IGNORECASE)
        return matches[0] if matches else None
    
    def _extract_date(self, text):
        """Extract date from text."""
        # Simple date extraction - expand as needed
        matches = re.findall(r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})', text)
        return matches[0] if matches else None
    
    def _extract_name(self, text):
        """Extract name from text."""
        # Simple name extraction - expand as needed
        matches = re.findall(r'name\s*[#:]*\s*([A-Za-z\s]+)', text, re.IGNORECASE)
        return matches[0].strip() if matches else None
    
    def _extract_phone(self, text):
        """Extract phone number from text."""
        # Simple phone extraction - expand as needed
        matches = re.findall(r'(\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4})', text)
        return matches[0] if matches else None
    
    def _extract_age(self, text):
        """Extract age from text."""
        # Simple age extraction - expand as needed
        matches = re.findall(r'age\s*[#:]*\s*(\d+)', text, re.IGNORECASE)
        return matches[0] if matches else None
    
    def validate_field(self, extracted_value, expected_value, field_rules):
        """Validate a single field based on its rules.
        
        Args:
            extracted_value: Value extracted from document
            expected_value: Value from reference data
            field_rules (dict): Validation rules for the field
            
        Returns:
            dict: Validation result with validity, action, and reason
        """
        # Handle missing values
        if extracted_value is None or pd.isna(extracted_value):
            return {"valid": False, "action": "manual", "reason": "Missing value"}
        
        if expected_value is None or pd.isna(expected_value):
            return {"valid": True, "action": "auto_correct", "reason": "No expected value to compare"}
        
        # Type-specific validation
        field_type = field_rules.get("type", "text")
        
        if field_type == "numeric":
            return self._validate_numeric(extracted_value, expected_value, field_rules)
        elif field_type == "date":
            return self._validate_date(extracted_value, expected_value, field_rules)
        elif field_type == "phone":
            return self._validate_phone(extracted_value, expected_value, field_rules)
        else:  # Text and other types
            return self._validate_text(extracted_value, expected_value, field_rules)
    
    def _validate_numeric(self, extracted, expected, field_rules):
        """Validate numeric fields."""
        try:
            ext_num = float(extracted)
            exp_num = float(expected)
            if abs(ext_num - exp_num) < 0.01:  # Allow small differences
                return {"valid": True, "action": "none", "reason": "Match"}
            else:
                action = "auto_correct" if field_rules.get("auto_correct", False) else "manual"
                return {"valid": False, "action": action, "reason": f"Value mismatch: {extracted} vs {expected}"}
        except ValueError:
            return {"valid": False, "action": "manual", "reason": "Invalid numeric format"}
    
    def _validate_date(self, extracted, expected, field_rules):
        """Validate date fields."""
        try:
            ext_date = pd.to_datetime(extracted)
            exp_date = pd.to_datetime(expected)
            if ext_date == exp_date:
                return {"valid": True, "action": "none", "reason": "Match"}
            else:
                action = "auto_correct" if field_rules.get("auto_correct", False) else "manual"
                return {"valid": False, "action": action, "reason": f"Date mismatch: {extracted} vs {expected}"}
        except ValueError:
            return {"valid": False, "action": "manual", "reason": "Invalid date format"}
    
    def _validate_phone(self, extracted, expected, field_rules):
        """Validate phone number fields."""
        # Normalize phone numbers by removing non-digit characters
        ext_clean = re.sub(r'\D', '', str(extracted))
        exp_clean = re.sub(r'\D', '', str(expected))
        
        if ext_clean == exp_clean:
            return {"valid": True, "action": "none", "reason": "Match"}
        else:
            action = "auto_correct" if field_rules.get("auto_correct", False) else "manual"
            return {"valid": False, "action": action, "reason": f"Phone mismatch: {extracted} vs {expected}"}
    
    def _validate_text(self, extracted, expected, field_rules):
        """Validate text fields."""
        if str(extracted).strip().lower() == str(expected).strip().lower():
            return {"valid": True, "action": "none", "reason": "Match"}
        else:
            action = "auto_correct" if field_rules.get("auto_correct", False) else "manual"
            return {"valid": False, "action": action, "reason": f"Value mismatch: {extracted} vs {expected}"}
    
    def find_best_reference_match(self, extracted_data, data_type):
        """Find the best matching reference row for the extracted data.
        
        Args:
            extracted_data (dict): Data extracted from document
            data_type (str): Type of data ('payment' or 'pii')
            
        Returns:
            tuple: (best matching row, match score)
        """
        if self.reference_data is None or self.reference_data.empty:
            return None, 0
            
        best_match = None
        best_score = 0
        
        for _, row in self.reference_data.iterrows():
            score = 0
            max_score = len(self.validation_rules[data_type])
            
            for field in self.validation_rules[data_type]:
                extracted_value = extracted_data.get(field)
                expected_value = row.get(field)
                
                if extracted_value is not None and expected_value is not None:
                    if str(extracted_value).strip().lower() == str(expected_value).strip().lower():
                        score += 1
            
            if score > best_score:
                best_score = score
                best_match = row
        
        return best_match, best_score
    
    def process_document(self, image_path, data_type, reference_excel_path):
        """Process document and validate against reference data.
        
        Args:
            image_path (str): Path to document image
            data_type (str): Type of data ('payment' or 'pii')
            reference_excel_path (str): Path to reference Excel file
            
        Returns:
            dict: Processing results with validation details
        """
        # REMINDER: Add support for batch processing multiple documents
        try:
            # Load reference data
            self.load_reference_data(reference_excel_path)
            
            # Extract text from image
            extracted_text = self.extract_text_with_ocr(image_path)
            if not extracted_text.strip():
                return {
                    "status": "error",
                    "message": "No text could be extracted from the image",
                    "extracted_data": {}
                }
            
            # Extract structured data
            extracted_data = self.extract_structured_data(extracted_text, data_type)
            
            # Find the best matching reference row
            reference_row, match_score = self.find_best_reference_match(extracted_data, data_type)
            
            if reference_row is None:
                return {
                    "status": "error",
                    "message": "No matching reference record found",
                    "extracted_data": extracted_data
                }
            
            # Validate the data
            validation_results = {}
            for field in self.validation_rules[data_type]:
                extracted_value = extracted_data.get(field)
                expected_value = reference_row.get(field)
                field_rules = self.validation_rules[data_type][field]
                
                validation_results[field] = self.validate_field(
                    extracted_value, expected_value, field_rules
                )
            
            # Determine actions
            actions = {
                "auto_correct": [],
                "manual_review": [],
                "valid": []
            }
            
            for field, result in validation_results.items():
                if result["action"] == "auto_correct":
                    actions["auto_correct"].append(field)
                elif result["action"] == "manual":
                    actions["manual_review"].append(field)
                else:
                    actions["valid"].append(field)
            
            return {
                "status": "success",
                "match_score": match_score,
                "extracted_data": extracted_data,
                "reference_data": reference_row.to_dict(),
                "validation_results": validation_results,
                "actions": actions
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error processing document: {str(e)}",
                "extracted_data": {}
            }
    
    def generate_detailed_report(self, result, report_path):
        """Generate a detailed validation report.
        
        Args:
            result (dict): Processing results from process_document
            report_path (str): Path to save the report
        """
        try:
            with open(report_path, 'w') as f:
                f.write("DATA VALIDATION REPORT\n")
                f.write("=" * 50 + "\n\n")
                
                f.write(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Status: {result['status']}\n")
                
                if result['status'] == 'error':
                    f.write(f"Error: {result['message']}\n")
                    return
                
                f.write(f"Match Score: {result['match_score']}\n\n")
                
                f.write("EXTRACTED DATA:\n")
                for field, value in result['extracted_data'].items():
                    f.write(f"  {field}: {value}\n")
                
                f.write("\nREFERENCE DATA:\n")
                for field, value in result['reference_data'].items():
                    f.write(f"  {field}: {value}\n")
                
                f.write("\nVALIDATION RESULTS:\n")
                for field, validation in result['validation_results'].items():
                    status = "VALID" if validation['valid'] else "INVALID"
                    f.write(f"  {field}: {status} - {validation['reason']}\n")
                
                f.write("\nRECOMMENDED ACTIONS:\n")
                f.write(f"  Auto-correct fields: {', '.join(result['actions']['auto_correct']) or 'None'}\n")
                f.write(f"  Manual review needed: {', '.join(result['actions']['manual_review']) or 'None'}\n")
                f.write(f"  Valid fields: {', '.join(result['actions']['valid']) or 'None'}\n")
                
                # Add summary statistics
                total_fields = len(result['validation_results'])
                if total_fields > 0:
                    valid_fields = len(result['actions']['valid'])
                    auto_correct_fields = len(result['actions']['auto_correct'])
                    manual_fields = len(result['actions']['manual_review'])
                    
                    f.write("\nSUMMARY:\n")
                    f.write(f"  Total fields: {total_fields}\n")
                    f.write(f"  Valid fields: {valid_fields} ({valid_fields/total_fields*100:.1f}%)\n")
                    f.write(f"  Auto-correct fields: {auto_correct_fields} ({auto_correct_fields/total_fields*100:.1f}%)\n")
                    f.write(f"  Manual review fields: {manual_fields} ({manual_fields/total_fields*100:.1f}%)\n")
                    
        except Exception as e:
            print(f"Error generating report: {e}")


# Example usage
if __name__ == "__main__":
    validator = AdvancedDataValidator()
    
    # Process a payment document
    payment_result = validator.process_document(
        image_path="payment_document.png",
        data_type="payment",
        reference_excel_path="payment_reference.xlsx"
    )
    
    if payment_result["status"] == "success":
        validator.generate_detailed_report(payment_result, "payment_validation_report.txt")
    
    # Process a PII document
    pii_result = validator.process_document(
        image_path="pii_document.png",
        data_type="pii",
        reference_excel_path="pii_reference.xlsx"
    )
    
    if pii_result["status"] == "success":
        validator.generate_detailed_report(pii_result, "pii_validation_report.txt")
    
    print("Processing completed. Check reports for details.")