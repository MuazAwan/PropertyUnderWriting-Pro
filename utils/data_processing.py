import streamlit as st
import PyPDF2
import openai
import os
import numpy as np
from typing import Dict
import logging
from litellm import completion

def generate_response(query: str, context: Dict) -> str:
    """Generate response using OpenAI or Gemini API"""
    try:
        if not st.session_state.OPENAI_API_KEY and not st.session_state.GOOGLE_API_KEY:
            error_msg = "Please provide either OpenAI or Google API key in the sidebar."
            logging.error(error_msg)
            return error_msg

        system_message = """You are an expert real estate investment analyst assistant. 
        Analyze the provided property data and answer questions with:
        1. Specific numbers and calculations
        2. Market insights and trends
        3. Investment recommendations
        4. Risk analysis
        5. Improvement opportunities
        
        Format responses with:
        • Clear sections
        • Bullet points
        • Specific metrics
        • Actionable recommendations"""

        # Use OpenAI if available, otherwise use Gemini
        if st.session_state.OPENAI_API_KEY:
            model = "gpt-4"
            api_key = st.session_state.OPENAI_API_KEY
        else:
            model = "gemini/gemini-1.5-flash"  # Keep original model name
            api_key = st.session_state.GOOGLE_API_KEY
            
        response = completion(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"Property Analysis Data:\n{context}\n\nQuestion: {query}"}
            ],
            api_key=api_key
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        logging.error(f"Error generating response: {str(e)}")
        return f"I apologize, but I encountered an error: {str(e)}"

def display_chat_interface(metrics: Dict, analysis_results: Dict, openai_key: str):
    """Display and handle the chat interface"""
    try:
        # Initialize chat history
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Create enhanced context dictionary with merged data
        context = {
            # Financial Metrics
            "total_income": analysis_results.get("total_income", 0),
            "total_expenses": analysis_results.get("total_expenses", 0),
            "offer_price": analysis_results.get("offer_price", 0),
            "debt_service": analysis_results.get("debt_service", 0),
            "noi": metrics.get("NOI", 0),
            "cap_rate": metrics.get("Cap Rate (%)", 0),
            "cash_on_cash_return": metrics.get("Cash on Cash Return (%)", 0),
            "dscr": metrics.get("DSCR", 0),
            
            # Property Details
            "num_units": analysis_results.get("num_units", 0),
            "occupancy_rate": analysis_results.get("occupancy_rate", 0),
            "year_built": analysis_results.get("year_built", 0),
            "property_type": analysis_results.get("property_type", "Not provided"),
            
            # Market Analysis
            "market_rent": analysis_results.get("market_rent", 0),
            "submarket_trends": analysis_results.get("submarket_trends", ""),
            "employment_growth_rate": analysis_results.get("employment_growth_rate", 0),
            "crime_rate": analysis_results.get("crime_rate", 0),
            "school_ratings": analysis_results.get("school_ratings", 0),
            
            # Additional Income
            "parking_income": analysis_results.get("parking_income", 0),
            "laundry_income": analysis_results.get("laundry_income", 0),
            
            # Capital & Improvements
            "renovation_cost": analysis_results.get("renovation_cost", 0),
            "capex": analysis_results.get("capex", 0),
            
            # Calculated Metrics
            "expense_ratio": metrics.get("Expense Ratio (%)", 0),
            "price_per_unit": metrics.get("Price per Unit", 0),
            "breakeven_occupancy": metrics.get("Breakeven Occupancy (%)", 0)
        }

        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Handle user input
        user_query = st.chat_input("Ask a question about the property analysis:")
        
        if user_query:
            # Add user message to chat
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            
            # Generate and add response
            response = generate_response(user_query, context)
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            
            # Rerun to update the display
            st.rerun()

        # Add clear chat button
        if st.sidebar.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
            
    except Exception as e:
        st.error(f"Chat interface error: {str(e)}")
        logging.error(f"Chat interface error: {str(e)}")
