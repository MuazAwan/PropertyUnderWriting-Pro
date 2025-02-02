# UnderwritePro - AI-Powered Property Underwriting Application

## Overview
UnderwritePro is an advanced AI-driven real estate underwriting platform that automates financial analysis, risk assessment, and investment strategy generation. Built with Streamlit, it enables users to analyze property data, visualize key metrics, and generate institutional-grade reports. It supports multiple data formats, integrates with OpenAI's GPT-4 for insights, and provides an interactive chat interface for underwriting consultation.

## Features
- **Multi-File Data Extraction:** Supports Excel, CSV, and PDF file uploads.
- **Comprehensive Financial Metrics Calculation:** Includes NOI, Cap Rate, DSCR, Cash-on-Cash Return, and Breakeven Occupancy.
- **AI-Powered Insights:** Generates detailed property analysis, risk assessment, and investment potential evaluations using GPT-4.
- **Dynamic Visualization:** Generates bar, pie, and line charts with export capabilities.
- **Chat Interface:** AI-driven real estate investment assistant for interactive analysis.
- **Professional PDF Reports:** Exports underwriting results, insights, and visualizations into a formatted report.

## Installation
### Prerequisites
- Python 3.9+
- OpenAI API Key (for AI-powered insights)

### Steps
1. Clone the repository:
   ```sh
   git clone https://github.com/MuazAwan/Property-Deploy.git
   cd Property-Deploy
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Configure environment variables:
   - Create a `.env` file in the root directory and add your OpenAI API key:
     ```
     OPENAI_API_KEY=your_api_key_here
     ```
4. Launch the application:
   ```sh
   streamlit run app.py
   ```

## Configuration
- **OpenAI API Key:** Required for AI insights and chatbot functionality. Enter your key in the sidebar after launching the app.
- **File Uploads:** Ensure files are in supported formats (Excel, CSV, or PDF).
- **PDF Parsing:** Extracts numeric fields such as `total_income`, `offer_price`, etc.

## Usage
### Step 1: Data Input
- Upload financial statements or rent rolls (Excel, CSV, or PDF).
- Enter property details manually if needed.

### Step 2: Analysis
- Choose the type of analysis (General, Risk, Investment Potential).
- Click **Analyze** to compute financial metrics and generate AI insights.

### Step 3: Visualization & Reporting
- View interactive charts.
- Export analysis as a professional PDF report.

### Step 4: AI Consultation
- Use the chatbot to ask questions like:
  - "What’s the ROI if occupancy increases by 5%?"
  - "Suggest 3 strategies to improve NOI."

## File Structure
```
underwritepro/
├── app.py                 # Main application logic
├── utils/
│   ├── data_processing.py # Handles file parsing (Excel, CSV, PDF)
│   ├── calculations.py    # Computes financial metrics
│   ├── chatbot.py         # GPT-4-powered Q&A interface
│   ├── llm_analysis.py    # Generates AI insights and risk assessment
│   ├── visualization.py   # Creates charts and visualizations
├── .env                   # API key configuration (optional)
```

## Contributing
- Fork the repository.
- Create a feature branch:
  ```sh
  git checkout -b feature/new-module
  ```
- Commit changes and push to your branch.
- Submit a pull request with a detailed description.

## License
This project is licensed under the MIT License. See `LICENSE` for details.

_Disclaimer: This tool provides AI-generated recommendations. Always validate outputs with certified professionals._

