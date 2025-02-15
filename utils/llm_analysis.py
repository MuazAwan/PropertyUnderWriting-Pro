import google.generativeai as genai
import openai
import logging
from typing import Dict, Any, Union
import json
import streamlit as st
from litellm import completion

def generate_insights(
    data: Union[Dict[str, Any], str], 
    model: str,
    insight_type: str = "general"
) -> str:
    """Generate comprehensive insights using all available data"""
    try:
        # Gather all available data sources
        analysis_data = {
            # Basic Financial Metrics
            "total_income": data.get("total_income", 0),
            "total_expenses": data.get("total_expenses", 0),
            "offer_price": data.get("offer_price", 0),
            "noi": data.get("noi", 0),
            "cap_rate": data.get("cap_rate", 0),
            
            # Property Details
            "property_details": {
                "num_units": data.get("num_units", 0),
                "year_built": data.get("year_built", 0),
                "occupancy_rate": data.get("occupancy_rate", 0),
                "property_type": data.get("property_type", "Not provided"),
                "property_class": data.get("property_class", "Not provided")
            },
            
            # Loan Information (from PDF extraction)
            "loan_details": st.session_state.get("loan_details", {}),
            
            # Property Condition (from PDF extraction)
            "property_condition": st.session_state.get("property_condition", {}),
            
            # Market Analysis
            "market_analysis": {
                "market_rent": data.get("market_rent", 0),
                "submarket_trends": data.get("submarket_trends", ""),
                "employment_growth": data.get("employment_growth_rate", 0),
                "crime_rate": data.get("crime_rate", 0),
                "school_ratings": data.get("school_ratings", 0),
                **st.session_state.get("market_info", {})  # Additional market info from PDFs
            },
            
            # Legal and Compliance
            "legal_compliance": st.session_state.get("legal_compliance", {}),
            
            # Important Contract Clauses
            "important_clauses": st.session_state.get("important_clauses", {}),
            
            # Capital Expenditure
            "capital_improvements": {
                "renovation_cost": data.get("renovation_cost", 0),
                "capex": data.get("capex", 0),
                "recent_renovations": data.get("recent_renovations", ""),
                "planned_improvements": data.get("planned_improvements", "")
            },
            
            # Additional Income Sources
            "additional_income": {
                "parking_income": data.get("parking_income", 0),
                "laundry_income": data.get("laundry_income", 0),
                "other_income": data.get("other_income", 0)
            }
        }

        prompt = f"""As an expert real estate analyst, provide a comprehensive analysis of this property investment opportunity:

Property Overview:
{json.dumps(analysis_data["property_details"], indent=2)}

Financial Analysis:
- Total Income: ${analysis_data["total_income"]:,.2f}
- Total Expenses: ${analysis_data["total_expenses"]:,.2f}
- NOI: ${analysis_data["noi"]:,.2f}
- Cap Rate: {analysis_data["cap_rate"]:.2f}%

Loan Details:
{json.dumps(analysis_data["loan_details"], indent=2)}

Property Condition:
{json.dumps(analysis_data["property_condition"], indent=2)}

Market Analysis:
{json.dumps(analysis_data["market_analysis"], indent=2)}

Legal & Compliance:
{json.dumps(analysis_data["legal_compliance"], indent=2)}

Important Contract Terms:
{json.dumps(analysis_data["important_clauses"], indent=2)}

Capital Improvements:
{json.dumps(analysis_data["capital_improvements"], indent=2)}

Additional Income Sources:
{json.dumps(analysis_data["additional_income"], indent=2)}

Please provide a detailed analysis including:
1. Investment Potential
2. Risk Assessment
3. Market Position
4. Legal Considerations
5. Improvement Opportunities
6. Financial Projections
7. Specific Recommendations

Format the response in clear sections with bullet points and specific metrics."""

        # Use the model passed from app.py
        if model == "gpt-4":
            if not st.session_state.OPENAI_API_KEY:
                return "OpenAI API key is required for GPT-4 analysis"
            api_key = st.session_state.OPENAI_API_KEY
        else:
            if not st.session_state.GOOGLE_API_KEY:
                return "Google API key is required for Gemini analysis"
            model = "gemini/gemini-1.5-flash"
            api_key = st.session_state.GOOGLE_API_KEY
            
        logging.info(f"Making API call with model: {model}")
        
        response = completion(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert real estate investment analyst."},
                {"role": "user", "content": prompt}
            ],
            api_key=api_key,
            max_tokens=2048,
            temperature=0.1
        )
        
        if response and hasattr(response, 'choices') and response.choices:
            analysis = response.choices[0].message.content
            
            # Format the response
            formatted_analysis = f"""
# Property Investment Analysis Report

{analysis}

---
*This analysis is generated using AI assistance and should be reviewed by qualified professionals.*
"""
            return formatted_analysis
            
        else:
            error_msg = "Failed to generate analysis. Please try again."
            logging.error(error_msg)
            return error_msg

    except Exception as e:
        error_msg = f"Error in generate_insights: {str(e)}"
        logging.error(error_msg)
        return error_msg

def _extract_property_data(text: str, model: str) -> str:
    """Extract structured property data from text using OpenAI's API."""
    system_message = """You are a real estate data extraction expert. Extract numeric values from the document and return ONLY a JSON object like this example:
    {"total_income":150000,"total_expenses":80000,"offer_price":1500000}
    
    Possible fields to extract:
    - total_income (annual)
    - total_expenses (annual)
    - offer_price
    - debt_service (annual)
    - equity
    - capex
    - market_rent (monthly)
    - num_units
    - occupancy_rate
    - year_built
    - price_per_unit
    - average_in_place_rent (monthly)

    IMPORTANT RULES:
    1. Return ONLY the JSON object - no other text or formatting
    2. Remove all $ signs, commas, and % signs
    3. Convert all values to plain numbers
    4. Only include fields you find in the document
    5. Do not add any explanations or comments"""
    
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": text}
            ],
            temperature=0.1,
            max_tokens=1000
        )
        
        response_text = response.choices[0].message.content.strip()
        logging.info(f"Raw LLM response: {response_text}")
        
        # Clean up any potential formatting
        if "```" in response_text:
            # Extract content between backticks
            response_text = response_text.split("```")[1].strip()
            if response_text.startswith("json"):
                response_text = response_text[4:].strip()
        
        # Remove any whitespace and newlines
        response_text = response_text.replace("\n", "").replace(" ", "")
        
        # Validate JSON
        try:
            json.loads(response_text)
            return response_text
        except json.JSONDecodeError:
            logging.error(f"Invalid JSON after cleanup: {response_text}")
            return "{}"
            
    except Exception as e:
        logging.error(f"Error in property data extraction: {str(e)}")
        return "{}"
