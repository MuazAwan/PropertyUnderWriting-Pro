import pandas as pd
from typing import Optional, List, Dict, Union
import logging
import os
import json
import streamlit as st
from litellm import completion
import numpy as np
from PyPDF2 import PdfReader
import time
from .calculations import calculate_metrics


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def parse_file(
    uploaded_file,
    required_columns: Optional[List[str]] = None,
    optional_columns: Optional[List[str]] = None
) -> Dict[str, Union[pd.DataFrame, List[str], dict]]:
    """Parse and validate uploaded files (Excel, CSV, or PDF)."""
    try:
        logging.info(f"Processing file: {uploaded_file.name}")
        file_name = uploaded_file.name.lower()
        
        if file_name.endswith(".pdf"):
            result = _parse_pdf(uploaded_file)
            
            if result and "extracted_data" in result:
                extracted_data = result["extracted_data"]
                
                # Financial Metrics
                if "Financial Metrics" in extracted_data:
                    metrics = extracted_data["Financial Metrics"]
                    st.session_state["total_income"] = _safe_convert(metrics.get("total_income"), float, 0)
                    st.session_state["total_expenses"] = _safe_convert(metrics.get("total_expenses"), float, 0)
                    st.session_state["offer_price"] = _safe_convert(metrics.get("offer_price"), float, 0)
                    st.session_state["debt_service"] = _safe_convert(metrics.get("debt_service"), float, 0)
                    st.session_state["cash_on_cash_return"] = _safe_convert(metrics.get("cash_on_cash_return"), float, 0) * 100
                    st.session_state["cap_rate"] = _safe_convert(metrics.get("cap_rate"), float, 0) * 100
                
                # Property Details
                if "Property Details" in extracted_data:
                    details = extracted_data["Property Details"]
                    st.session_state["num_units"] = _safe_convert(details.get("num_units"), int, 0)
                    st.session_state["occupancy_rate"] = _safe_convert(details.get("occupancy_rate"), float, 0) * 100
                    st.session_state["year_built"] = _safe_convert(details.get("year_built"), int, 0)
                
                # Market Analysis
                if "Market Analysis" in extracted_data:
                    market = extracted_data["Market Analysis"]
                    st.session_state["market_rent"] = _safe_convert(market.get("market_rent"), float, 0)
                    st.session_state["submarket_trends"] = str(market.get("submarket_trends", ""))
                    st.session_state["employment_growth_rate"] = _safe_convert(market.get("employment_growth_rate"), float, 0) * 100
                    st.session_state["crime_rate"] = _safe_convert(market.get("crime_rate"), float, 0)
                    st.session_state["school_ratings"] = _safe_convert(market.get("school_ratings"), float, 5)
                
                # Capital & Improvements
                if "Capital & Improvements" in extracted_data:
                    capital = extracted_data["Capital & Improvements"]
                    st.session_state["renovation_cost"] = _safe_convert(capital.get("renovation_cost"), float, 0)
                    st.session_state["capex"] = _safe_convert(capital.get("capex_budget"), float, 0)
                
                # Income Details
                if "Income Details" in extracted_data:
                    income = extracted_data["Income Details"]
                    st.session_state["parking_income"] = _safe_convert(income.get("parking_income"), float, 0)
                    st.session_state["laundry_income"] = _safe_convert(income.get("laundry_income"), float, 0)
                
                logging.info("Successfully auto-filled session state with extracted data")
                
            return result
            
        elif file_name.endswith(".xlsx"):
            # Process Excel file
            data = _parse_excel(uploaded_file)
            if not data.empty:
                return {
                    "data": data,
                    "file_name": file_name,
                    "file_type": "excel",
                    "extracted_data": data.iloc[0].to_dict()
                }
            
        return {"extracted_data": {}, "data": pd.DataFrame()}
            
    except Exception as e:
        logging.error(f"Error processing file: {str(e)}")
        return {"extracted_data": {}, "data": pd.DataFrame()}

def _parse_file_with_retry(prompt: str, max_retries: int = 3, initial_delay: int = 5) -> dict:
    """Helper function to retry API calls with exponential backoff"""
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                sleep_time = initial_delay * (4 ** attempt)
                logging.info(f"Waiting {sleep_time} seconds before retry {attempt + 1}")
                time.sleep(sleep_time)
            
            response = completion(
                model="gemini/gemini-1.5-flash",
                messages=[{
                    "role": "user", 
                    "content": f"""You are a real estate data extraction expert. 
                    {prompt}
                    
                    Focus on extracting exact numbers and values.
                    Return only valid JSON with the specified fields."""
                }],
                api_key=st.session_state.GOOGLE_API_KEY,
                max_tokens=2048,
                temperature=0.1,
                request_timeout=45
            )
            
            if hasattr(response, 'choices') and response.choices:
                content = response.choices[0].message.content
                if content:
                    try:
                        content = content.strip()
                        if content.startswith("```json"):
                            content = content[7:-3]
                        return json.loads(content)
                    except json.JSONDecodeError as je:
                        logging.error(f"JSON parse error: {je}\nContent: {content}")
                        if attempt < max_retries - 1:
                            continue
                        return {"error": "Failed to parse JSON response"}
            
            logging.warning(f"Empty response on attempt {attempt + 1}")
            if attempt < max_retries - 1:
                continue
            return {"error": "Empty response"}
            
        except Exception as e:
            error_msg = str(e)
            logging.warning(f"Attempt {attempt + 1} failed: {error_msg}")
            
            if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                if attempt < max_retries - 1:
                    sleep_time = initial_delay * (4 ** attempt)
                    logging.info(f"Rate limit hit. Waiting {sleep_time} seconds...")
                    time.sleep(sleep_time)
                    continue
                return {"error": "Rate limit exceeded"}
            
            if attempt < max_retries - 1:
                continue
            return {"error": f"API error: {error_msg}"}
    
    return {"error": "Max retries reached"}

def _parse_pdf(uploaded_file) -> dict:
    """Enhanced PDF parsing with table and text extraction"""
    try:
        # Read PDF bytes and extract text
        reader = PdfReader(uploaded_file)
        text = "\n".join([page.extract_text() for page in reader.pages])
        
        # Enhanced extraction prompt for property and loan analysis
        extraction_prompt = f"""Extract key real estate metrics and important information from this document:
        {text}

        Return JSON format with these fields (use null if not found):
        
        Financial Metrics:
        - total_income (annual number)
        - total_expenses (annual number)
        - offer_price (number)
        - debt_service (annual number)
        - noi (annual number)
        - cap_rate (percentage)
        - cash_on_cash_return (percentage)
        - dscr (number)
        
        Property Details:
        - property_type (text)
        - year_built (number)
        - num_units (number)
        - occupancy_rate (percentage)
        - average_unit_size (number in sqft)
        - property_class (text: A, B, or C)
        
        Market Analysis:
        - market_rent (number per unit)
        - submarket_trends (text)
        - employment_growth_rate (percentage)
        - crime_rate (text or number)
        - school_ratings (text or number)
        
        Capital & Improvements:
        - renovation_cost (number)
        - capex_budget (number)
        - recent_renovations (text)
        - planned_improvements (text)
        
        Income Details:
        - parking_income (annual number)
        - laundry_income (annual number)
        - other_income (annual number)
        - rent_growth_rate (percentage)
        
        Expense Details:
        - property_tax (annual number)
        - insurance (annual number)
        - utilities (annual number)
        - maintenance (annual number)
        - management_fee (percentage)
        
        Additional Information:
        - lease_terms (text)
        - tenant_mix (text)
        - amenities (text)
        - financial_highlights (text)
        
        Loan Details:
        - loan_amount (number)
        - interest_rate (percentage)
        - loan_term (years)
        - amortization_period (years)
        - prepayment_penalty (text)
        - loan_type (text)
        - lender_requirements (text)
        
        Property Condition:
        - building_condition (text)
        - deferred_maintenance (text)
        - recent_repairs (text)
        - needed_repairs (text)
        - environmental_issues (text)
        
        Market Information:
        - local_market_trends (text)
        - comparable_properties (text)
        - demographic_trends (text)
        - development_plans (text)
        
        Legal & Compliance:
        - zoning_restrictions (text)
        - permits_required (text)
        - code_violations (text)
        - regulatory_issues (text)
        
        Important Clauses:
        - key_lease_terms (text)
        - special_conditions (text)
        - restrictions (text)
        - exit_clauses (text)
        
        Rules:
        1. Convert all currencies to numbers without symbols
        2. Convert percentages to decimal (e.g. 95% → 0.95)
        3. Return ONLY valid JSON
        4. Use null for missing values"""
        
        # Use OpenAI if available, otherwise use Gemini
        if st.session_state.OPENAI_API_KEY:
            model = "gpt-4"
            api_key = st.session_state.OPENAI_API_KEY
        else:
            model = "gemini/gemini-1.5-flash"
            api_key = st.session_state.GOOGLE_API_KEY
            
        logging.info(f"Making API call with model: {model}")
        
        response = completion(
            model=model,
            messages=[{"role": "user", "content": extraction_prompt}],
            api_key=api_key
        )
        
        # Process response and store additional context
        if response and hasattr(response, 'choices') and response.choices:
            content = response.choices[0].message.content
            if content:
                try:
                    # Clean the content
                    content = content.strip()
                    if "```json" in content:
                        content = content.split("```json")[1].split("```")[0]
                    elif "```" in content:
                        content = content.split("```")[1]
                    
                    content = content.strip()
                    logging.info(f"Cleaned content: {content}")
                    
                    # Parse JSON
                    extracted_data = json.loads(content)
                    
                    # Store additional context in session state
                    if isinstance(extracted_data, dict):
                        if "Loan Details" in extracted_data:
                            st.session_state["loan_details"] = extracted_data["Loan Details"]
                        if "Property Condition" in extracted_data:
                            st.session_state["property_condition"] = extracted_data["Property Condition"]
                        if "Market Information" in extracted_data:
                            st.session_state["market_info"] = extracted_data["Market Information"]
                        if "Legal & Compliance" in extracted_data:
                            st.session_state["legal_compliance"] = extracted_data["Legal & Compliance"]
                        if "Important Clauses" in extracted_data:
                            st.session_state["important_clauses"] = extracted_data["Important Clauses"]
                        
                        return {"extracted_data": extracted_data, "data": pd.DataFrame([extracted_data])}
                    
                except json.JSONDecodeError as je:
                    logging.error(f"JSON parse error: {je}\nContent: {content}")
                    return {"extracted_data": {}, "data": pd.DataFrame()}
                    
        logging.error("Invalid or empty response")
        return {"extracted_data": {}, "data": pd.DataFrame()}
            
    except Exception as e:
        logging.error(f"PDF parsing failed: {str(e)}")
        return {"extracted_data": {}, "data": pd.DataFrame()}

def _parse_excel(uploaded_file) -> pd.DataFrame:
    """Parse Excel file and extract all data for semantic analysis."""
    try:
        excel_file = pd.ExcelFile(uploaded_file)
        all_sheets_data = {}
        
        for sheet_name in excel_file.sheet_names:
            logging.info(f"Processing sheet: {sheet_name}")
            df = pd.read_excel(excel_file, sheet_name=sheet_name)
            
            # Clean column names
            df = _clean_column_names(df)
            
            # Convert DataFrame to structured dictionary
            sheet_data = {}
            
            # Process each row
            for idx, row in df.iterrows():
                row_data = {}
                for col in df.columns:
                    value = row[col]
                    # Handle different data types
                    if pd.isna(value):
                        row_data[col] = None
                    elif isinstance(value, (np.int64, np.float64)):
                        row_data[col] = float(value) if float(value).is_integer() else float(value)
                    else:
                        row_data[col] = str(value)
                
                sheet_data[f"row_{idx}"] = row_data
            
            all_sheets_data[sheet_name] = sheet_data
            
        # Create final JSON structure
        extracted_data = {
            "file_info": {
                "filename": uploaded_file.name,
                "total_sheets": len(excel_file.sheet_names),
                "sheet_names": excel_file.sheet_names
            },
            "sheets": all_sheets_data
        }
        
        logging.info(f"Successfully extracted data from Excel: {len(all_sheets_data)} sheets")
        
        # Create DataFrame for compatibility
        result_df = pd.DataFrame([extracted_data])
        return result_df
        
    except Exception as e:
        logging.error(f"Error in Excel parsing: {str(e)}")
        return pd.DataFrame()

def _clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize column names."""
    # Remove unnamed columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    
    # Clean column names
    df.columns = df.columns.str.strip().str.lower()
    df.columns = df.columns.str.replace(' ', '_')
    df.columns = df.columns.str.replace('[^a-z0-9_]', '', regex=True)
    
    # Remove empty rows and columns
    df = df.dropna(how='all').dropna(axis=1, how='all')
    
    return df

def _merge_processed_data(processed_data_list: List[Dict]) -> pd.DataFrame:
    """Merge data from multiple processed files."""
    try:
        # Initialize empty DataFrame for merged data
        merged_df = pd.DataFrame()
        
        for file_data in processed_data_list:
            current_df = file_data['data']
            
            if merged_df.empty:
                merged_df = current_df
            else:
                # Merge based on common columns
                common_cols = list(set(merged_df.columns) & set(current_df.columns))
                if common_cols:
                    # Use fillna to keep existing values when merging
                    merged_df = pd.concat([merged_df, current_df], axis=0).groupby(level=0).first()
                else:
                    # If no common columns, just append
                    merged_df = pd.concat([merged_df, current_df], axis=1)
            
        # Fill NaN values with 0
        merged_df = merged_df.fillna(0)
        
        return merged_df
            
    except Exception as e:
        logging.error(f"Error merging data: {str(e)}")
        return pd.DataFrame()

def display_extracted_data(processed_data_list: List[Dict]):
    """Display extracted data from all files in a readable format."""
    st.write("## Extracted Data from All Files")
    
    for file_data in processed_data_list:
        try:
            file_name = file_data.get('file_name', 'Unknown File')
            data_df = file_data.get('data', pd.DataFrame())
            
            st.write(f"### Data from {file_name}")
            
            if not data_df.empty:
                # Handle different data types
                if isinstance(data_df.iloc[0], (np.float64, np.int64, float, int)):
                    # Handle numeric data directly
                    st.write(f"Value: {float(data_df.iloc[0])}")
                    continue
                    
                if file_name.endswith('.pdf'):
                    # Display PDF extracted data
                    extracted_data = file_data.get('extracted_data', {})
                    if extracted_data:
                        st.write("#### Extracted Values:")
                        for key, value in extracted_data.items():
                            if isinstance(value, (np.float64, np.int64)):
                                value = float(value)
                            st.write(f"- {key.replace('_', ' ').title()}: {value}")
                else:
                    # Display Excel/CSV data
                    if isinstance(data_df.iloc[0], dict) and 'content' in data_df.iloc[0]:
                        content = data_df.iloc[0]['content']
                        if isinstance(content, dict):
                            for sheet_name, sheet_data in content.items():
                                st.write(f"#### Sheet: {sheet_name}")
                                # Create a DataFrame for better display
                                rows_data = []
                                if isinstance(sheet_data, dict):
                                    for row_data in sheet_data.values():
                                        if isinstance(row_data, dict):
                                            rows_data.append(row_data)
                                if rows_data:
                                    sheet_df = pd.DataFrame(rows_data)
                                    st.dataframe(sheet_df)
                                st.write("---")
                    else:
                        # Display the DataFrame directly
                        st.dataframe(data_df)
                        
        except Exception as e:
            logging.error(f"Error displaying data for {file_name}: {str(e)}")
            st.error(f"Error displaying data: {str(e)}")

def _normalize_data(raw_data: dict) -> dict:
    """Standardize data formats across sources"""
    normalized = {}
    
    # Numeric fields with currency
    currency_fields = ['total_income', 'total_expenses', 'offer_price', 'debt_service']
    for field in currency_fields:
        if raw_data.get(field):
            try:
                value = str(raw_data[field]).replace('$', '').replace(',', '')
                normalized[field] = float(value)
            except:
                normalized[field] = 0
    
    # Percentage fields
    percentage_fields = ['occupancy_rate']
    for field in percentage_fields:
        if raw_data.get(field):
            try:
                value = str(raw_data[field]).replace('%', '')
                normalized[field] = float(value) / 100
            except:
                normalized[field] = 0
    
    # Integer fields
    integer_fields = ['year_built', 'num_units']
    for field in integer_fields:
        if raw_data.get(field):
            try:
                normalized[field] = int(float(str(raw_data[field])))
            except:
                normalized[field] = 0
    
    # Text fields
    text_fields = ['property_type', 'recent_renovations', 'lease_terms', 'financial_highlights']
    for field in text_fields:
        if raw_data.get(field):
            normalized[field] = str(raw_data[field]).strip()
    
    return normalized

def _safe_convert(value, convert_type, default=0):
    """Safely convert values to specified type."""
    try:
        if value is None:
            return default
        if convert_type == float:
            # Handle percentage conversion
            if isinstance(value, str) and '%' in value:
                value = value.replace('%', '')
                return float(value) / 100
            return float(value)
        if convert_type == int:
            return int(float(value))
        if convert_type == str:
            return str(value) if value is not None else ""
        return default
    except (ValueError, TypeError):
        return default

def _has_minimum_data() -> bool:
    """Check if we have minimum required data for analysis"""
    # Check if we have any meaningful data from files or user input
    has_file_data = len(st.session_state.get("processed_files", [])) > 0
    has_income = st.session_state.get("total_income", 0) > 0
    has_expenses = st.session_state.get("total_expenses", 0) > 0
    has_units = st.session_state.get("num_units", 0) > 0
    has_occupancy = st.session_state.get("occupancy_rate", 0) > 0
    
    # Return True if we have either file data or at least some key metrics
    return has_file_data or (has_income or has_expenses or has_units or has_occupancy)
