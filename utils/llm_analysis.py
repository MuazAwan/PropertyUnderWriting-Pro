import openai
import logging
from typing import Dict, Any, Union
import json

def generate_insights(
    data: Union[Dict[str, Any], str], 
    model: str = "gpt-4", 
    insight_type: str = "general"
) -> str:
    """Generate insights or extract data using OpenAI's API."""
    try:
        if not openai.api_key:
            error_msg = "OpenAI API key not set. Please provide your API key in the sidebar."
            logging.error(error_msg)
            return error_msg
        
        # Handle PDF extraction separately
        if insight_type == "extract_values":
            if not isinstance(data, str):
                raise ValueError("PDF extraction requires text input")
            logging.info("Starting PDF data extraction")
            return _extract_property_data(data, model)
            
        # Calculate additional metrics for analysis
        calculated_metrics = {
            "noi": data.get("total_income", 0) - data.get("total_expenses", 0),
            "total_other_income": data.get("parking_income", 0) + data.get("laundry_income", 0),
            "effective_gross_income": data.get("total_income", 0) + 
                                    data.get("parking_income", 0) + 
                                    data.get("laundry_income", 0),
            "operating_expense_ratio": (data.get("total_expenses", 0) / data.get("total_income", 0) * 100) 
                                     if data.get("total_income", 0) > 0 else 0,
            "price_per_sf": data.get("offer_price", 0) / data.get("total_square_feet", 1) 
                           if data.get("total_square_feet", 0) > 0 else 0,
            "gross_rent_multiplier": data.get("offer_price", 0) / 
                                    (data.get("total_income", 0) * 12) if data.get("total_income", 0) > 0 else 0,
            "debt_coverage_ratio": data.get("noi", 0) / 
                                  data.get("debt_service", 1) if data.get("debt_service", 0) > 0 else 0
        }
        
        # Combine original data with calculated metrics
        analysis_data = {**data, **calculated_metrics}
        
        prompt = f"""Analyze these real estate metrics as an experienced underwriter:

Property Overview:
- Year Built: {data.get('year_built', 'Not provided')} (Age: {2024 - data.get('year_built', 2024)} years)
- Number of Units: {data.get('num_units', 'Not provided')}
- Price per Unit: ${data.get('offer_price', 0) / data.get('num_units', 1):,.2f}
- Market Rent per Unit: ${data.get('market_rent', 0):,.2f}
- Current Occupancy: {data.get('occupancy_rate', 0)}%

Key Performance Indicators:
- NOI per Unit: ${data.get('noi', 0) / data.get('num_units', 1):,.2f}
- Expense Ratio: {(data.get('total_expenses', 0) / data.get('total_income', 1) * 100):.1f}%
- DSCR: {data.get('noi', 0) / data.get('debt_service', 1) if data.get('debt_service', 0) > 0 else 0:.2f}
- Cash on Cash Return: {data.get('cash_on_cash_return', 0):.1f}%

Financial Metrics:
- Offer Price: ${data.get('offer_price', 0):,.2f}
- Total Income: ${data.get('total_income', 0):,.2f}
- Total Expenses: ${data.get('total_expenses', 0):,.2f}
- NOI: ${data.get('noi', 0):,.2f}
- Cap Rate: {data.get('cap_rate', 0):.2f}%
- Debt Service: ${data.get('debt_service', 0):,.2f}
- Equity Required: ${data.get('equity', 0):,.2f}

Additional Income:
- Parking Income: ${data.get('parking_income', 0):,.2f}
- Laundry Income: ${data.get('laundry_income', 0):,.2f}
- Total Other Income: ${data.get('parking_income', 0) + data.get('laundry_income', 0):,.2f}

Market Analysis:
- Crime Rate: {data.get('crime_rate', 0)}
- School Rating: {data.get('school_ratings', 0)}/10
- Employment Growth: {data.get('employment_growth_rate', 0)}%
- Submarket Trends: {data.get('submarket_trends', 'Not provided')}

Please provide a detailed analysis with proper formatting:
1. All dollar amounts as $X,XXX.XX
2. All percentages as XX.XX%
3. Include all property metrics in the analysis
4. Provide specific recommendations based on the property's characteristics

Please provide recommendations in these categories:
1. Immediate Actions
2. Short-term Improvements (0-6 months)
3. Long-term Strategy (6+ months)
4. Risk Mitigation Steps
5. Exit Strategy Considerations

For each recommendation:
- Estimated cost/benefit
- Implementation timeline
- Expected impact on NOI
"""

        if insight_type == "improvement":
            prompt += "\nFocus heavily on sections 6 and 7, with detailed improvement strategies."
        elif insight_type == "risk_analysis":
            prompt += "\nProvide expanded analysis of section 5, with detailed risk mitigation strategies."
        elif insight_type == "investment_potential":
            prompt += "\nEmphasize sections 1, 2, and 7, with detailed investment return analysis."

        system_prompt = """You are an expert real estate underwriter with 20+ years of experience in multifamily property analysis. 
        Provide detailed, professional analysis using industry standard metrics and terminology. 
        Support all conclusions with data-driven insights.
        Be direct and specific in your recommendations.
        Format the response in a clear, professional structure using markdown."""

        logging.info(f"Generating {insight_type} underwriting analysis")
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,  # Reduced for more consistent, analytical responses
            max_tokens=2500,  # Increased for more comprehensive analysis
            top_p=0.9,
            presence_penalty=0.3,
            frequency_penalty=0.3
        )
        
        analysis = response.choices[0].message.content
        
        # Format the response with a professional header
        formatted_analysis = f"""
# Property Underwriting Analysis Report

{analysis}

---
*This analysis is generated using AI assistance and should be reviewed by qualified professionals. All recommendations should be independently verified.*
"""
        return formatted_analysis
        
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
