import streamlit as st
import PyPDF2
import openai
import os
import numpy as np
from typing import Dict
import logging

def generate_response(query: str, context: Dict) -> str:
    """Generate response using OpenAI API"""
    try:
        if not openai.api_key:
            error_msg = "OpenAI API key not set. Please provide your API key in the sidebar."
            logging.error(error_msg)
            return error_msg

        system_message = """You are an expert real estate investment analyst assistant. 
        Analyze the property data and provide detailed, actionable insights.
        Focus on specific numbers and calculations.
        Provide recommendations with expected ROI where possible.
        Format your response with clear sections and bullet points."""

        # Format context as a string with proper formatting
        context_str = f"""
        Property Analysis Summary:
        
        Financial Metrics:
        • NOI: ${context.get('noi', 0):,.2f}
        • Cap Rate: {context.get('cap_rate', 0):.2f}%
        • Cash on Cash Return: {context.get('cash_on_cash_return', 0):.2f}%
        • DSCR: {context.get('dscr', 0):.2f}
        
        Income & Expenses:
        • Total Income: ${context.get('total_income', 0):,.2f}
        • Total Expenses: ${context.get('total_expenses', 0):,.2f}
        • Offer Price: ${context.get('offer_price', 0):,.2f}
        • Debt Service: ${context.get('debt_service', 0):,.2f}
        
        Property Details:
        • Number of Units: {context.get('num_units', 0)}
        • Occupancy Rate: {context.get('occupancy_rate', 0)}%
        • Market Rent: ${context.get('market_rent', 0):,.2f}
        • CapEx Budget: ${context.get('capex', 0):,.2f}
        """

        logging.info(f"Generating response for query: {query}")
        logging.info(f"Context: {context_str}")

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"{context_str}\n\nQuestion: {query}"}
            ],
            temperature=0.7,
            max_tokens=1000
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

        # Create context dictionary
        context = {
            "total_income": analysis_results.get("total_income", 0),
            "total_expenses": analysis_results.get("total_expenses", 0),
            "offer_price": analysis_results.get("offer_price", 0),
            "debt_service": analysis_results.get("debt_service", 0),
            "noi": metrics.get("noi", 0),
            "cap_rate": metrics.get("cap_rate", 0),
            "cash_on_cash_return": metrics.get("cash_on_cash_return", 0),
            "dscr": metrics.get("dscr", 0),
            "num_units": analysis_results.get("num_units", 0),
            "occupancy_rate": analysis_results.get("occupancy_rate", 0),
            "market_rent": analysis_results.get("market_rent", 0),
            "capex": analysis_results.get("capex", 0),
            "submarket_trends": metrics.get("submarket_trends", "Not provided"),
            "employment_growth_rate": metrics.get("employment_growth_rate", 0),
            "crime_rate": metrics.get("crime_rate", 0),
            "school_ratings": metrics.get("school_ratings", 0)
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
