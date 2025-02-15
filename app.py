import streamlit as st
import pandas as pd
from property1.utils.data_processing import parse_file, _has_minimum_data
from property1.utils.calculations import calculate_metrics
from property1.utils.llm_analysis import generate_insights
from property1.utils.visualization import plot_metrics
from property1.utils.chatbot import display_chat_interface

import os
import matplotlib.pyplot as plt
from fpdf import FPDF
from PIL import Image
import logging
import openai
import gc
import sys


# Define GEMINI_AVAILABLE before it's used
GEMINI_AVAILABLE = True  # Set this based on whether Gemini functionality is available

# Set page config
st.set_page_config(
    page_title="UnderwritePro",
    page_icon="📊",
    layout="wide"
)

# For render.com deployment
port = int(os.environ.get("PORT", 8080))

def _safe_convert(value, convert_type, default=0):
    """Safely convert values to specified type."""
    try:
        if value is None or value == '':
            return default
        if isinstance(value, str):
            # Remove any non-numeric characters except decimal point
            value = ''.join(c for c in value if c.isdigit() or c == '.')
        return convert_type(float(value))
    except (ValueError, TypeError):
        return default

# Add this near the top of the app, before any analysis functions
def initialize_api_keys():
    """Initialize API keys from session state or user input"""
    if 'OPENAI_API_KEY' not in st.session_state:
        st.session_state.OPENAI_API_KEY = None
    if 'GOOGLE_API_KEY' not in st.session_state:
        st.session_state.GOOGLE_API_KEY = None
    
    with st.sidebar:
        st.write("## API Keys")
        # Add model selection
        model_options = ["GPT-4 (for smaller inputs)"]
        if GEMINI_AVAILABLE:
            model_options.append("Gemini (for larger inputs)")
            
        st.session_state.selected_model = st.radio(
            "Select AI Model:",
            model_options,
            help="Choose GPT-4 for precise analysis of smaller datasets, or Gemini for larger datasets"
        )
        
        if st.session_state.selected_model == "GPT-4 (for smaller inputs)":
            openai_key = st.text_input(
                "Enter your OpenAI API key:",
                type="password",
                help="Get your API key from https://platform.openai.com/account/api-keys",
                value=st.session_state.OPENAI_API_KEY if st.session_state.OPENAI_API_KEY else ""
            )
            if openai_key:
                st.session_state.OPENAI_API_KEY = openai_key
                st.success("OpenAI API key set successfully!")
        elif GEMINI_AVAILABLE:
            google_key = st.text_input(
                "Enter your Google API key:",
                type="password",
                help="Get your API key from Google AI Studio",
                value=st.session_state.GOOGLE_API_KEY if st.session_state.GOOGLE_API_KEY else ""
            )
            if google_key:
                st.session_state.GOOGLE_API_KEY = google_key
                st.success("Google API key set successfully!")

# Add this right after the app title
initialize_api_keys()

# Update the API key validation message
if not st.session_state.OPENAI_API_KEY and not st.session_state.GOOGLE_API_KEY:
    st.warning("Please enter either an OpenAI API key or a Google API key in the sidebar to use AI analysis features.")

# Initialize session state for inputs
basic_inputs = [
    "offer_price", 
    "total_income", 
    "total_expenses", 
    "equity", 
    "debt_service",
    "occupancy_rate"
]
additional_inputs = [
    "market_rent", "cash_on_cash_return", "projected_cap_rate_at_sale",
    "breakeven_occupancy", "year_built", "num_units", "unit_mix",
    "occupancy_rate_trends", "market_growth_rate", "price_per_unit",
    "average_in_place_rent", "submarket_trends", "employment_growth_rate",
    "crime_rate", "school_ratings", "renovation_cost", "capex",
    "holding_period", "rent_variation", "expense_variation",
    "parking_income", "laundry_income", "tenant_type"
]

# Initialize session states with proper default values
for input_name in basic_inputs + additional_inputs:
    if input_name not in st.session_state:
        # Set appropriate default values based on input type
        if input_name == "occupancy_rate":
            st.session_state[input_name] = 0.0  # or any default percentage
        elif input_name in ["unit_mix", "submarket_trends", "tenant_type"]:
            st.session_state[input_name] = ""
        else:
            st.session_state[input_name] = 0.0

# Initialize session state variables if not exists
if "data" not in st.session_state:
    st.session_state.update({
        "data": None,
        "metrics": None,
        "total_income": 0.0,
        "total_expenses": 0.0,
        "offer_price": 0.0,
        "debt_service": 0.0,
        "num_units": 0,
        "occupancy_rate": 0.0,
        "market_rent": 0.0,
        "renovation_cost": 0.0,
        "capex": 0.0,
        "parking_income": 0.0,
        "laundry_income": 0.0,
        # ... other fields ...
    })

if "insight_type" not in st.session_state:
    st.session_state["insight_type"] = "general"

if "chart_type" not in st.session_state:
    st.session_state["chart_type"] = "bar"

# Add this with your other session state initializations at the top
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

# Streamlit app starts here
st.title("UnderwritePro")
st.write("Provide detailed inputs for a comprehensive analysis and actionable insights.")

# Add at the top of app.py
gc.collect()  # Force garbage collection

# Update file paths to be absolute
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Replace the existing file uploader with this
uploaded_files = st.file_uploader(
    "Upload files (Excel, CSV, or PDF)", 
    type=["xlsx", "csv", "pdf"],
    accept_multiple_files=True
)


# Update the PDF parsing error handling
if uploaded_files:
    extracted_data_list = []
    
    for uploaded_file in uploaded_files:
        try:
            # Parse file and extract data
            result = parse_file(uploaded_file, None, None)
            
            if uploaded_file.name.endswith('.pdf'):
                extracted_data = result.get("extracted_data", {})
                if extracted_data:
                    extracted_data_list.append({
                        "filename": uploaded_file.name,
                        "data": extracted_data
                    })
            else:
                # Handle Excel/CSV data
                data = result.get("data")
                if not data.empty:
                    extracted_data_list.append({
                        "filename": uploaded_file.name,
                        "data": data.to_dict('records')[0]
                    })
        
        except Exception as e:
            if "PyCryptodome is required for AES algorithm" in str(e):
                st.warning("PyCryptodome is required for AES algorithm. Please install it using: pip install pycryptodome")
            else:
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")
            logging.error(f"File processing error: {str(e)}")
            continue
    
    # Merge extracted data
    if extracted_data_list:
        with st.expander("View Extracted Data from All Files"):
            for extracted_item in extracted_data_list:
                st.subheader(f"Data from {extracted_item['filename']}")
                st.json(extracted_item['data'])
        
        # Merge data giving priority to non-zero values
        merged_data = {}
        for item in extracted_data_list:
            for key, value in item['data'].items():
                if key not in merged_data or (value != 0 and merged_data[key] == 0):
                    merged_data[key] = value
        
        # Update session state with merged values
        fields_to_update = {
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
            "unit_mix": "unit_mix",
            "price_per_unit": "price_per_unit",
            "average_in_place_rent": "average_in_place_rent",
            "submarket_trends": "submarket_trends",
            "employment_growth_rate": "employment_growth_rate",
            "crime_rate": "crime_rate",
            "school_ratings": "school_ratings",
            "cash_on_cash_return": "cash_on_cash_return",
            "projected_cap_rate_at_sale": "projected_cap_rate_at_sale",
            "breakeven_occupancy": "breakeven_occupancy",
            "renovation_cost": "renovation_cost",
            "tenant_type": "tenant_type",
            "parking_income": "parking_income",
            "laundry_income": "laundry_income",
            "rent_variation": "rent_variation",
            "expense_variation": "expense_variation"
        }
        
        for file_key, state_key in fields_to_update.items():
            if file_key in merged_data:
                st.session_state[state_key] = merged_data[file_key]
        
        st.success("Data extracted and merged successfully! Please review the pre-filled values.")

# Inputs - Basic Metrics
st.header("Basic Metrics")
st.session_state["offer_price"] = st.number_input(
    "Offer Price ($):",
    min_value=0.0,
    max_value=1000000000.0,
    value=float(st.session_state.get("offer_price", 0.0)),
    step=1000.0,
    format="%.2f"
)
st.session_state["total_income"] = st.number_input(
    "Total Income ($):",
    min_value=0.0,
    max_value=1000000000.0,
    value=float(st.session_state.get("total_income", 0.0)),
    step=1000.0,
    format="%.2f"
)
st.session_state["total_expenses"] = st.number_input(
    "Total Expenses ($):",
    min_value=0.0,
    max_value=1000000000.0,
    value=float(st.session_state.get("total_expenses", 0.0)),
    step=1000.0,
    format="%.2f"
)
st.session_state["equity"] = st.number_input(
    "Enter Equity ($) (Optional)", min_value=0.0, value=st.session_state["equity"], step=1000.0
)
st.session_state["debt_service"] = st.number_input(
    "Debt Service ($):",
    min_value=0.0,
    max_value=1000000000.0,
    value=float(st.session_state.get("debt_service", 0.0)),
    step=1000.0,
    format="%.2f"
)

# Add this with other basic metrics inputs
st.session_state["occupancy_rate"] = st.slider(
    "Current Occupancy Rate (%)", 
    min_value=0.0, 
    max_value=100.0, 
    value=float(st.session_state["occupancy_rate"]),
    step=1.0
)

# Inputs - Property Overview
with st.expander("Property Overview (Optional)"):
    st.session_state["year_built"] = st.slider(
        "Year Built (Optional)", min_value=1800, max_value=2100, value=int(st.session_state["year_built"])
    )
    st.session_state["num_units"] = st.slider(
        "Number of Units (Optional)", min_value=0, max_value=1000, value=int(st.session_state["num_units"])
    )
    st.session_state["unit_mix"] = st.text_input(
        "Unit Mix (Optional)", value=st.session_state["unit_mix"], help="e.g., 1B1B: 10 units, 2B2B: 5 units"
    )
    st.session_state["price_per_unit"] = st.number_input(
        "Price Per Unit ($) (Optional)", min_value=0.0, value=st.session_state["price_per_unit"], step=100.0
    )
    st.session_state["average_in_place_rent"] = st.number_input(
        "Average In-Place Rent ($) (Optional)", min_value=0.0, value=st.session_state["average_in_place_rent"], step=50.0
    )

# Inputs - Market Analysis
with st.expander("Market Analysis (Optional)"):
    st.session_state["submarket_trends"] = st.text_area(
        "Submarket Trends (Optional)", value=st.session_state["submarket_trends"], 
        help="Provide details on local market conditions and comparable properties."
    )
    st.session_state["employment_growth_rate"] = st.slider(
        "Employment Growth Rate (%) (Optional)", min_value=-100, max_value=100, value=int(st.session_state["employment_growth_rate"])
    )
    st.session_state["crime_rate"] = st.slider(
        "Crime Rate (%) (Optional)", 
        min_value=0, 
        max_value=100, 
        value=_safe_convert(st.session_state.get("crime_rate", 0), int, 0)  # Safe conversion
    )
    st.session_state["school_ratings"] = st.slider(
        "School Ratings (1-10) (Optional)", 
        min_value=1, 
        max_value=10, 
        value=_safe_convert(st.session_state.get("school_ratings", 5), int, 5)  # Safe conversion
    )

# Inputs - Financial Metrics
with st.expander("Financial Metrics (Optional)"):
    st.session_state["cash_on_cash_return"] = st.slider(
        "Cash-on-Cash Return (%) (Optional)", min_value=-100, max_value=100, value=int(st.session_state["cash_on_cash_return"])
    )
    st.session_state["projected_cap_rate_at_sale"] = st.slider(
        "Projected Cap Rate at Sale (%) (Optional)", min_value=0.0, max_value=100.0, value=st.session_state["projected_cap_rate_at_sale"], step=0.1
    )
    st.session_state["breakeven_occupancy"] = st.slider(
        "Breakeven Occupancy Rate (%) (Optional)", min_value=0, max_value=100, value=int(st.session_state["breakeven_occupancy"])
    )
    st.session_state["renovation_cost"] = st.number_input(
        "Total Renovation Costs ($) (Optional)", 
        min_value=0.0,  # float
        value=float(st.session_state.get("renovation_cost", 0.0)),  # convert to float
        step=1000.0,  # float
        format="%.2f"  # ensure float format
    )
    st.session_state["capex"] = st.number_input(
        "Capital Expenditures (CapEx) ($) (Optional)", 
        min_value=0.0,  # float
        value=float(st.session_state.get("capex", 0.0)),  # convert to float
        step=1000.0,  # float
        format="%.2f"  # ensure float format
    )

# Inputs - Tenant and Revenue Analysis
with st.expander("Tenant and Revenue Analysis (Optional)"):
    st.session_state["tenant_type"] = st.selectbox(
        "Primary Tenant Type (Optional)", 
        ["Family-Oriented", "Students", "Working Professionals", "Retirees"], 
        index=0
    )
    
    st.session_state["laundry_income"] = st.number_input(
        "Laundry Income ($) (Optional)",
        min_value=0.0,
        max_value=1000000.0,
        value=float(st.session_state.get("laundry_income", 0.0)),  # Convert to float
        step=50.0,
        format="%.2f"
    )

# Sensitivity Analysis
with st.expander("Sensitivity Analysis (Optional)"):
    st.session_state["rent_variation"] = st.slider(
        "Rent Variation (%)", min_value=-50, max_value=50, value=0, step=5, help="Simulate changes in rent levels."
    )
    st.session_state["expense_variation"] = st.slider(
        "Expense Variation (%)", min_value=-50, max_value=50, value=0, step=5, help="Simulate changes in expense levels."
    )

# Add analysis type selector before the Analyze button
insight_type = st.selectbox(
    "Select Analysis Type",
    ["general", "improvement", "risk_analysis", "investment_potential"],
    index=0,
    help="Choose the type of analysis you want to generate"
)

# Chart Type
chart_type = st.selectbox(
    "Select Chart Type", ["bar", "pie", "line"], key="chart_type"
)

# Function to plot metrics and save the graph
def plot_metrics(metrics, chart_type="bar", save_path="chart.png"):
    if not metrics or all(value == 0 for value in metrics.values()):
        st.warning("No meaningful data to plot.")
        return

    try:
        fig, ax = plt.subplots(figsize=(10, 6))  # Adjust figure size for better readability

        if chart_type == "bar":
            ax.bar(metrics.keys(), metrics.values(), color='skyblue', edgecolor='black')
            ax.set_title("Financial Metrics", fontsize=16, fontweight='bold')
            ax.set_ylabel("Value ($)", fontsize=12)
            ax.set_xlabel("Metrics", fontsize=12)
            ax.set_xticks(range(len(metrics.keys())))
            ax.set_xticklabels(metrics.keys(), rotation=45, ha="right", fontsize=10)
            ax.grid(axis='y', linestyle='--', alpha=0.7)

        elif chart_type == "pie":
            if len(metrics) > 10:
                st.warning("Too many metrics for a pie chart. Consider using a bar or line chart.")
                return
            ax.pie(metrics.values(), labels=metrics.keys(), autopct='%1.1f%%', startangle=90)
            ax.set_title("Financial Metrics Distribution", fontsize=16, fontweight='bold')

        elif chart_type == "line":
            ax.plot(list(metrics.keys()), list(metrics.values()), marker='o', linestyle='-', linewidth=2)
            ax.set_title("Financial Metrics Over Time", fontsize=16, fontweight='bold')
            ax.set_ylabel("Value ($)", fontsize=12)
            ax.set_xlabel("Metrics", fontsize=12)
            ax.set_xticks(range(len(metrics.keys())))
            ax.set_xticklabels(metrics.keys(), rotation=45, ha="right", fontsize=10)
            ax.grid(axis='both', linestyle='--', alpha=0.7)

        fig.tight_layout()  # Adjust layout to prevent label clipping
        fig.savefig(save_path, format="png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        st.image(save_path, caption="Generated Chart")
    except Exception as e:
        st.error(f"Error generating chart: {e}")

# Function to generate PDF with graph
def save_to_pdf_with_graph(metrics, insights, chart_image_path, file_name="UnderwritePro_Output.pdf"):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Title
    pdf.set_font("Arial", style="B", size=16)
    pdf.cell(200, 10, txt="UnderwritePro Output Report", ln=True, align="C")
    pdf.ln(10)  # Add space

    # Metrics Section
    pdf.set_font("Arial", style="B", size=14)
    pdf.cell(200, 10, txt="Calculated Metrics:", ln=True)
    pdf.set_font("Arial", size=12)
    for key, value in metrics.items():
        pdf.cell(0, 10, txt=f"{key}: {value}", ln=True)

    # Insights Section
    if insights:
        pdf.ln(10)
        pdf.set_font("Arial", style="B", size=14)
        pdf.cell(200, 10, txt="Generated Insights:", ln=True)
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, txt=insights)

    # Add Graph
    if chart_image_path:
        with Image.open(chart_image_path) as img:
            width, height = img.size
            aspect_ratio = height / width

        max_width = 180
        img_width = max_width
        img_height = max_width * aspect_ratio

        max_height = 200
        if img_height > max_height:
            img_height = max_height
            img_width = max_height / aspect_ratio

        pdf.ln(10)
        pdf.set_font("Arial", style="B", size=14)
        pdf.cell(200, 10, txt="Visualization:", ln=True)
        pdf.image(chart_image_path, x=10, y=None, w=img_width, h=img_height)

    pdf.output(file_name)
    return file_name

# Analyze button
if st.button("Analyze"):
    if not st.session_state.OPENAI_API_KEY and not st.session_state.GOOGLE_API_KEY:
        st.error("Please enter either an OpenAI API key or a Google API key in the sidebar to perform analysis.")
    else:
        try:
            # Before generating insights
            if any(value < 0 for value in [
                st.session_state["offer_price"],
                st.session_state["total_income"],
                st.session_state["total_expenses"]
            ]):
                st.error("Please ensure all basic metrics are non-negative values.")
            # Validate percentages
            if not 0 <= st.session_state["occupancy_rate"] <= 100:
                st.warning("Occupancy rate should be between 0 and 100%")

            # First check if we have the required data
            if not st.session_state["total_income"] or not st.session_state["total_expenses"] or not st.session_state["offer_price"]:
                st.error("Please provide the required basic metrics (Income, Expenses, and Offer Price)")
            else:
                # Create base data dictionary with scalar values in a single row DataFrame
                data_dict = {
                    "Income": st.session_state["total_income"],
                    "Expenses": st.session_state["total_expenses"],
                    "offer_price": st.session_state["offer_price"],
                    "debt_service": st.session_state["debt_service"],
                    "equity": st.session_state["equity"],
                    "capex": st.session_state["capex"],
                    "market_rent": st.session_state["market_rent"],
                    "num_units": st.session_state["num_units"],
                    "occupancy_rate": st.session_state["occupancy_rate"],
                    "year_built": st.session_state["year_built"],
                    "price_per_unit": st.session_state["price_per_unit"],
                    "average_in_place_rent": st.session_state["average_in_place_rent"],
                    # Add these important metrics for better analysis
                    "unit_mix": st.session_state["unit_mix"],
                    "submarket_trends": st.session_state["submarket_trends"],
                    "employment_growth_rate": st.session_state["employment_growth_rate"],
                    "crime_rate": st.session_state["crime_rate"],
                    "school_ratings": st.session_state["school_ratings"],
                    "renovation_cost": st.session_state["renovation_cost"],
                    "breakeven_occupancy": st.session_state["breakeven_occupancy"],
                    "projected_cap_rate_at_sale": st.session_state["projected_cap_rate_at_sale"],
                    "cash_on_cash_return": st.session_state["cash_on_cash_return"],
                    # Add missing fields
                    "tenant_type": st.session_state["tenant_type"],
                    "parking_income": st.session_state["parking_income"],
                    "laundry_income": st.session_state["laundry_income"],
                    "rent_variation": st.session_state["rent_variation"],
                    "expense_variation": st.session_state["expense_variation"],
                    "holding_period": st.session_state["holding_period"]
                }
                
                # Create DataFrame with a single row
                st.session_state["data"] = pd.DataFrame([data_dict])
                
                # Calculate metrics
                additional_inputs_dict = {key: st.session_state[key] for key in additional_inputs}
                st.session_state["metrics"] = calculate_metrics(
                    st.session_state["data"], 
                    st.session_state["offer_price"], 
                    additional_inputs_dict
                )
                
                # Display results
                st.header("Analysis Results")
                
                # Show metrics
                with st.expander("View Calculated Metrics", expanded=True):
                    st.json(st.session_state["metrics"])

                # Before generating insights
                analysis_data = {
                    **st.session_state["metrics"],
                    # Basic Metrics
                    "offer_price": st.session_state["offer_price"],
                    "total_income": st.session_state["total_income"],
                    "total_expenses": st.session_state["total_expenses"],
                    "equity": st.session_state["equity"],
                    "debt_service": st.session_state["debt_service"],
                    
                    # Market Analysis
                    "submarket_trends": st.session_state["submarket_trends"],
                    "employment_growth_rate": st.session_state["employment_growth_rate"],
                    "crime_rate": st.session_state["crime_rate"],
                    "school_ratings": st.session_state["school_ratings"],
                    "occupancy_rate": st.session_state["occupancy_rate"],
                    
                    # Financial Metrics
                    "cash_on_cash_return": st.session_state["cash_on_cash_return"],
                    "projected_cap_rate_at_sale": st.session_state["projected_cap_rate_at_sale"],
                    "breakeven_occupancy": st.session_state["breakeven_occupancy"],
                    "renovation_cost": st.session_state["renovation_cost"],
                    "capex": st.session_state["capex"],
                    
                    # Tenant and Revenue Analysis
                    "tenant_type": st.session_state["tenant_type"],
                    "parking_income": st.session_state["parking_income"],
                    "laundry_income": st.session_state["laundry_income"],
                    
                    # Sensitivity Analysis
                    "rent_variation": st.session_state["rent_variation"],
                    "expense_variation": st.session_state["expense_variation"],
                    
                    # Ensure basic calculated values are included
                    "noi": st.session_state["metrics"].get("NOI", 0),
                    "cap_rate": st.session_state["metrics"].get("Cap Rate (%)", 0),
                    "dscr": st.session_state["metrics"].get("DSCR", 0)
                }

                # Generate insights with complete data
                insights = generate_insights(
                    analysis_data,
                    model="gpt-4" if st.session_state.selected_model == "GPT-4 (for smaller inputs)" else "gemini/gemini-1.5-flash",
                    insight_type=insight_type
                )
                
                if insights and not insights.startswith("Error"):
                    st.markdown(insights)
                else:
                    st.error(f"Failed to generate insights: {insights}")

                # Display visualization
                if st.session_state["metrics"]:
                    st.subheader("Metrics Visualization")
                    plot_metrics(st.session_state["metrics"], chart_type=chart_type)

        except Exception as e:
            st.error(f"Error during analysis: {str(e)}")
            logging.error(f"Analysis error: {str(e)}", exc_info=True)

# Export to PDF
if st.button("Export to PDF"):
    try:
        if st.session_state["metrics"]:
            chart_path = "chart.png"
            plot_metrics(st.session_state["metrics"], chart_type=st.session_state["chart_type"], save_path=chart_path)

            # Create complete analysis data for PDF export
            pdf_analysis_data = {
                **st.session_state["metrics"],
                "crime_rate": st.session_state["crime_rate"],
                "school_ratings": st.session_state["school_ratings"],
                "employment_growth_rate": st.session_state["employment_growth_rate"],
                "submarket_trends": st.session_state["submarket_trends"],
                "occupancy_rate": st.session_state["occupancy_rate"]
            }

            insights_text = generate_insights(
                pdf_analysis_data,
                model="gpt-4" if st.session_state.selected_model == "GPT-4 (for smaller inputs)" else "vertex_ai/gemini-1.5-flash",
                insight_type=insight_type
            ) if (st.session_state.OPENAI_API_KEY or st.session_state.GOOGLE_API_KEY) else "Insights require a valid API key."

            pdf_file = save_to_pdf_with_graph(st.session_state["metrics"], insights_text, chart_path)

            st.success(f"PDF generated successfully: {pdf_file}")
            with open(pdf_file, "rb") as f:
                st.download_button("Download PDF", f, file_name=pdf_file)
        else:
            st.error("No metrics to export. Perform analysis first.")
    except Exception as e:
        st.error(f"Failed to generate PDF: {e}")

# Chat Interface
st.header("Chat with AI Assistant")
st.write("Ask questions about the property analysis and get detailed insights.")

if st.session_state.get("metrics") and (st.session_state.OPENAI_API_KEY or st.session_state.GOOGLE_API_KEY):
    # Get analysis results from session state
    analysis_results = {
        "total_income": st.session_state.get("total_income", 0),
        "total_expenses": st.session_state.get("total_expenses", 0),
        "offer_price": st.session_state.get("offer_price", 0),
        "debt_service": st.session_state.get("debt_service", 0),
        "capex": st.session_state.get("capex", 0),
        "market_rent": st.session_state.get("market_rent", 0),
        "occupancy_rate": st.session_state.get("occupancy_rate", 0),
        "num_units": st.session_state.get("num_units", 0),
        # Add calculated metrics
        "noi": st.session_state["total_income"] - st.session_state["total_expenses"],
        "cap_rate": ((st.session_state["total_income"] - st.session_state["total_expenses"]) / 
                     st.session_state["offer_price"]) * 100 if st.session_state["offer_price"] > 0 else 0,
        "cash_flow": st.session_state["total_income"] - st.session_state["total_expenses"] - 
                     st.session_state["debt_service"],
        # Add additional relevant fields
        "tenant_type": st.session_state.get("tenant_type", ""),
        "parking_income": st.session_state.get("parking_income", 0),
        "laundry_income": st.session_state.get("laundry_income", 0),
        "rent_variation": st.session_state.get("rent_variation", 0),
        "expense_variation": st.session_state.get("expense_variation", 0),
        "total_other_income": st.session_state.get("parking_income", 0) + 
                             st.session_state.get("laundry_income", 0),
        "total_income": st.session_state.get("total_income", 0) + 
                        st.session_state.get("parking_income", 0) + 
                        st.session_state.get("laundry_income", 0),
        # Add missing market analysis fields
        "crime_rate": st.session_state.get("crime_rate", 0),
        "school_ratings": st.session_state.get("school_ratings", 0),
        "employment_growth_rate": st.session_state.get("employment_growth_rate", 0),
        "submarket_trends": st.session_state.get("submarket_trends", ""),
        
        # Add missing property details
        "year_built": st.session_state.get("year_built", 0),
        "num_units": st.session_state.get("num_units", 0),
        "market_rent": st.session_state.get("market_rent", 0),
        "price_per_unit": st.session_state.get("price_per_unit", 0),
        
        # Add missing financial metrics
        "renovation_cost": st.session_state.get("renovation_cost", 0),
        "breakeven_occupancy": st.session_state.get("breakeven_occupancy", 0),
        "projected_cap_rate_at_sale": st.session_state.get("projected_cap_rate_at_sale", 0),
        "cash_on_cash_return": st.session_state.get("cash_on_cash_return", 0),
        # Add missing calculated metrics
        "noi_per_unit": st.session_state["metrics"].get("NOI", 0) / max(st.session_state.get("num_units", 1), 1),
        "expense_ratio": (st.session_state.get("total_expenses", 0) / max(st.session_state.get("total_income", 1), 1)) * 100,
        "dscr": st.session_state["metrics"].get("DSCR", 0)
    }
    
    # Display chat interface
    display_chat_interface(
        metrics=st.session_state["metrics"],
        analysis_results=analysis_results,
        openai_key=st.session_state.OPENAI_API_KEY or st.session_state.GOOGLE_API_KEY
    )
else:
    st.info("Please analyze a property first to use the chat feature.")

# Add a divider
st.markdown("---")

# Add this after imports
def initialize_environment():
    """Initialize environment and handle errors"""
    try:
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Add the project root to Python path
        project_root = os.path.abspath(os.path.dirname(__file__))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
            
    except Exception as e:
        st.error(f"Error initializing environment: {str(e)}")
        logging.error(f"Environment initialization error: {str(e)}")

# Call initialization
initialize_environment()
