import pandas as pd
from typing import Optional, List, Dict, Union
import logging
import PyPDF2
import os
import json

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def parse_file(
    uploaded_file,
    required_columns: Optional[List[str]] = None,
    optional_columns: Optional[List[str]] = None
) -> Dict[str, Union[pd.DataFrame, List[str], dict]]:
    """Parse and validate uploaded files (Excel, CSV, or PDF)."""
    if uploaded_file is None:
        raise ValueError("No file provided")
    
    # Validate file type
    file_name = uploaded_file.name.lower()
    if not any(file_name.endswith(ext) for ext in [".pdf", ".xlsx", ".csv"]):
        raise ValueError("Unsupported file format. Please upload a .xlsx, .csv, or .pdf file.")

    try:
        logging.info(f"Processing file: {file_name}")
        extracted_data = {}
        
        if file_name.endswith(".pdf"):
            # Extract data from PDF
            extracted_result = _parse_pdf(uploaded_file)
            extracted_data = extracted_result.get("extracted_data", {})
            
            # Create DataFrame with matching field names
            data = pd.DataFrame({
                "total_income": [extracted_data.get("total_income", 0)],
                "total_expenses": [extracted_data.get("total_expenses", 0)],
                "offer_price": [extracted_data.get("offer_price", 0)],
                "debt_service": [extracted_data.get("debt_service", 0)],
                "equity": [extracted_data.get("equity", 0)],
                "capex": [extracted_data.get("capex", 0)],
                "market_rent": [extracted_data.get("market_rent", 0)],
                "num_units": [extracted_data.get("num_units", 0)],
                "occupancy_rate": [extracted_data.get("occupancy_rate", 0)],
                "year_built": [extracted_data.get("year_built", 0)],
                "price_per_unit": [extracted_data.get("price_per_unit", 0)],
                "average_in_place_rent": [extracted_data.get("average_in_place_rent", 0)]
            })
            
            # Return result with complete DataFrame
            return {
                "data": data,
                "extracted_data": extracted_data,
                "missing_columns": [],
                "detected_columns": list(data.columns)
            }
        elif file_name.endswith(".xlsx"):
            data = _parse_excel(uploaded_file)
        else:  # CSV file
            data = pd.read_csv(uploaded_file)
        
        # Only validate columns for Excel and CSV files
        if not file_name.endswith(".pdf"):
            result = _validate_and_process_data(
                data, 
                required_columns or [], 
                optional_columns or []
            )
            result['extracted_data'] = extracted_data
            return result
        
    except Exception as e:
        logging.error(f"Error processing file {file_name}: {str(e)}")
        raise ValueError(f"Error processing file: {str(e)}")

def _parse_pdf(uploaded_file) -> dict:
    """Parse PDF file and extract relevant property information."""
    try:
        # Extract text using PyPDF2
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text_content = "".join(page.extract_text() for page in pdf_reader.pages)
        logging.info("PDF text extracted successfully")
        
        # Use LLM to extract data
        from utils.llm_analysis import generate_insights
        extracted_json = generate_insights(text_content, model="gpt-4", insight_type="extract_values")
        logging.info(f"Raw extracted JSON: {extracted_json}")
        
        try:
            # Parse and validate JSON
            extracted_data = json.loads(extracted_json)
            logging.info(f"Parsed JSON data: {extracted_data}")
            
            # Clean and validate each value
            cleaned_data = {}
            field_mapping = {
                "total_income": "total_income",
                "total_expenses": "total_expenses",
                "offer_price": "offer_price",
                "debt_service": "debt_service",
                "equity": "equity",
                "capex": "capex",
                "market_rent": "market_rent",
                "num_units": "num_units",
                "occupancy_rate": "occupancy_rate",
                "year_built": "year_built",
                "price_per_unit": "price_per_unit",
                "average_in_place_rent": "average_in_place_rent"
            }
            
            # Process each field
            for json_key, app_key in field_mapping.items():
                if json_key in extracted_data:
                    value = extracted_data[json_key]
                    try:
                        if isinstance(value, str):
                            value = value.replace('$', '').replace(',', '').replace('%', '')
                        cleaned_value = float(value)
                        if cleaned_value != 0:  # Only include non-zero values
                            cleaned_data[app_key] = cleaned_value
                            logging.info(f"Extracted {app_key}: {cleaned_value}")
                    except (ValueError, TypeError) as e:
                        logging.warning(f"Could not convert {json_key}: {value} - {str(e)}")
            
            if not cleaned_data:
                logging.warning("No valid data extracted from PDF")
                return {"extracted_data": {}}
                
            logging.info(f"Final cleaned data: {cleaned_data}")
            return {"extracted_data": cleaned_data}
            
        except json.JSONDecodeError as e:
            logging.error(f"JSON parsing error: {str(e)}")
            return {"extracted_data": {}}
            
    except Exception as e:
        logging.error(f"Error in PDF parsing: {str(e)}")
        return {"extracted_data": {}}

def _parse_excel(uploaded_file) -> pd.DataFrame:
    """Parse Excel file, handling multiple sheets."""
    excel_file = pd.ExcelFile(uploaded_file)
    if len(excel_file.sheet_names) > 1:
        sheet_name = excel_file.sheet_names[0]
        logging.info(f"Multiple sheets found. Using first sheet: {sheet_name}")
    return pd.read_excel(uploaded_file)

def _validate_and_process_data(
    data: pd.DataFrame,
    required_columns: List[str],
    optional_columns: List[str]
) -> Dict[str, Union[pd.DataFrame, List[str]]]:
    """Validate and process the DataFrame."""
    # Check required columns for non-PDF files
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns and required_columns:
        msg = f"Required columns missing: {missing_columns}. Please ensure your Excel/CSV file contains these columns."
        logging.error(msg)
        raise ValueError(msg)

    # Detect optional columns
    detected_columns = [col for col in optional_columns if col in data.columns]
    logging.info(f"Detected optional columns: {detected_columns}")

    # Clean data
    data = _clean_dataframe(data, optional_columns)
    return {
        "data": data,
        "missing_columns": missing_columns,
        "detected_columns": detected_columns,
    }

def _clean_dataframe(data: pd.DataFrame, optional_columns: List[str]) -> pd.DataFrame:
    """Clean and prepare DataFrame for analysis."""
    # Fill NaN values
    data = data.fillna(0)

    # Convert numeric columns
    for col in data.columns:
        if data[col].dtype == "object":
            try:
                data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0)
            except Exception as e:
                logging.debug(f"Could not convert column {col} to numeric: {str(e)}")

    # Add missing optional columns
    for col in optional_columns:
        if col not in data.columns:
            data[col] = 0

    return data
