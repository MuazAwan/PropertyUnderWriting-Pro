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
        
        prompt = f"""Analyze these real estate metrics as an experienced underwriter:\n{json.dumps(analysis_data, indent=2)}\n\n
        Provide a detailed underwriting analysis with the following structure:

        1. Executive Summary
           - Property overview (Year built: {data.get('year_built')}, Units: {data.get('num_units')})
           - Total price: ${data.get('offer_price', 0):,.2f}
           - Price per unit: ${data.get('price_per_unit', 0):,.2f}
           - NOI: ${calculated_metrics['noi']:,.2f}
           - Investment highlights
           - Major risk factors
           - Overall recommendation

        2. Financial Analysis
           - Current NOI and projected stabilized NOI
           - Cap rate: Current vs Market ({data.get('projected_cap_rate_at_sale')}%)
           - Cash flow analysis (Including debt service of ${data.get('debt_service', 0):,.2f})
           - Cash-on-cash return target: {data.get('cash_on_cash_return')}%
           - Operating expense ratio: {calculated_metrics['operating_expense_ratio']:.1f}%
           - Breakeven occupancy analysis: {data.get('breakeven_occupancy')}%

        3. Market Analysis
           - Submarket trends: {data.get('submarket_trends', 'Not provided')}
           - Employment growth: {data.get('employment_growth_rate')}%
           - Crime rate index: {data.get('crime_rate')}
           - School ratings: {data.get('school_ratings')}/10
           - Rent comparables and growth potential

        4. Property Assessment
           - Unit mix breakdown: {data.get('unit_mix', 'Not provided')}
           - Current occupancy vs market
           - Average in-place rent: ${data.get('average_in_place_rent', 0):,.2f}
           - Tenant profile: {data.get('tenant_type', 'Not provided')}
           - Additional income sources:
             * Parking: ${data.get('parking_income', 0):,.2f}
             * Laundry: ${data.get('laundry_income', 0):,.2f}

        5. Risk Assessment
           - Sensitivity Analysis:
             * Rent variation: {data.get('rent_variation')}%
             * Expense variation: {data.get('expense_variation')}%
           - Market risks
           - Property-specific risks
           - Financial risks
           - Management risks
           - Mitigation strategies

        6. Value-Add Opportunities
           - Renovation budget: ${data.get('renovation_cost', 0):,.2f}
           - CapEx requirements: ${data.get('capex', 0):,.2f}
           - Revenue enhancement strategies
           - Expense reduction opportunities

        7. Investment Recommendations
           - Pricing analysis
           - Deal structure (Equity: ${data.get('equity', 0):,.2f})
           - Holding period: {data.get('holding_period', 'Not specified')} years
           - Exit strategy
           - Action items

        Additional Analysis Requirements:
        - Compare all metrics to industry standards
        - Provide specific, actionable recommendations
        - Include quantitative justification for conclusions
        - Factor in all additional income sources
        - Consider renovation ROI and timing
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

