def calculate_metrics(data, offer_price, additional_inputs=None):
    """
    Calculate financial metrics such as NOI, Cap Rate, Cash on Cash Return, DSCR, and others.
    
    Args:
        data: DataFrame containing financial data (Income, Expenses, etc.).
        offer_price: Purchase price of the property.
        additional_inputs: Dictionary containing additional data like equity, debt service, and other parameters.
    
    Returns:
        dict: A dictionary containing calculated financial metrics.
    """
    try:
        # Extract basic data
        income = data.get('Income', []).sum()
        expenses = data.get('Expenses', []).sum()
        
        # Add additional income sources
        parking_income = additional_inputs.get("parking_income", 0)
        laundry_income = additional_inputs.get("laundry_income", 0)
        total_other_income = parking_income + laundry_income
        
        # Update total income to include additional sources
        total_income = income + total_other_income
        
        # Calculate NOI with total income
        noi = total_income - expenses

        # Update other calculations
        cap_rate = (noi / offer_price) * 100 if offer_price > 0 else 0
        equity = additional_inputs.get("equity", 0)
        debt_service = additional_inputs.get("debt_service", 0)
        cash_on_cash_return = (noi / equity) * 100 if equity > 0 else 0
        dscr = noi / debt_service if debt_service > 0 else 0
        breakeven_occupancy = (expenses / total_income) * 100 if total_income > 0 else 0
        
        # Compile metrics
        metrics = {
            "NOI": noi,
            "Cap Rate (%)": cap_rate,
            "Cash on Cash Return (%)": cash_on_cash_return,
            "DSCR": dscr,
            "Breakeven Occupancy (%)": breakeven_occupancy,
            "Rent Per Unit ($)": (income / data.get('Number of Units', 0)) if data.get('Number of Units', 0) > 0 else 0,
            "Expense Per Unit ($)": (expenses / data.get('Number of Units', 0)) if data.get('Number of Units', 0) > 0 else 0,
            "Rent Gap ($)": 0,
            "Rent Gap (%)": 0,
            "Projected Cap Rate at Sale (%)": 0,
            "Market Growth Rate (%)": 0,
            "Parking Income ($)": parking_income,
            "Laundry Income ($)": laundry_income,
            "Other Income ($)": total_other_income,
            "Adjusted NOI ($)": noi,
            "Adjusted Income ($)": total_income,
            "Adjusted Expenses ($)": expenses,
            "Gross Rent Multiplier": offer_price / (total_income * 12) if total_income > 0 else 0
        }
        
        return metrics
    except Exception as e:
        raise ValueError(f"Error calculating metrics: {e}")
