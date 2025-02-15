import logging
import pandas as pd

def calculate_metrics(data, offer_price, additional_inputs=None):
    """Calculate financial metrics from extracted data and user inputs."""
    try:
        # Initialize additional_inputs if None
        additional_inputs = additional_inputs or {}
        
        # Helper function to safely get numeric values
        def safe_get_value(data_dict, key, default=0):
            value = data_dict.get(key, default)
            if isinstance(value, pd.Series):
                return float(value.iloc[0]) if not value.empty else default
            return float(value) if value is not None else default
        
        # Extract income and expenses with safe conversion
        total_income = safe_get_value(data, 'total_income') or safe_get_value(additional_inputs, 'total_income')
        total_expenses = safe_get_value(data, 'total_expenses') or safe_get_value(additional_inputs, 'total_expenses')
        
        # Get additional income sources
        parking_income = safe_get_value(additional_inputs, 'parking_income')
        laundry_income = safe_get_value(additional_inputs, 'laundry_income')
        total_other_income = parking_income + laundry_income
        
        # Update total income with additional sources
        total_income += total_other_income
        
        # Get property details with safe conversion
        num_units = int(safe_get_value(data, 'num_units') or safe_get_value(additional_inputs, 'num_units'))
        occupancy_rate = safe_get_value(data, 'occupancy_rate') or safe_get_value(additional_inputs, 'occupancy_rate')
        
        # Calculate NOI
        noi = total_income - total_expenses
        
        # Calculate key metrics with safe conversion
        offer_price_value = float(offer_price) if isinstance(offer_price, (int, float, str)) else safe_get_value({'offer_price': offer_price}, 'offer_price')
        cap_rate = (noi / offer_price_value) * 100 if offer_price_value > 0 else 0
        
        equity = safe_get_value(additional_inputs, 'equity')
        debt_service = safe_get_value(additional_inputs, 'debt_service')
        
        # Calculate financial ratios
        cash_on_cash_return = (noi / equity) * 100 if equity > 0 else 0
        dscr = noi / debt_service if debt_service > 0 else 0
        breakeven_occupancy = (total_expenses / total_income) * 100 if total_income > 0 else 0
        
        # Per unit calculations
        rent_per_unit = total_income / num_units if num_units > 0 else 0
        expense_per_unit = total_expenses / num_units if num_units > 0 else 0
        
        # Return comprehensive metrics dictionary
        return {
            "NOI": round(noi, 2),
            "Cap Rate (%)": round(cap_rate, 2),
            "Cash on Cash Return (%)": round(cash_on_cash_return, 2),
            "DSCR": round(dscr, 2),
            "Breakeven Occupancy (%)": round(breakeven_occupancy, 2),
            "Rent Per Unit ($)": round(rent_per_unit, 2),
            "Expense Per Unit ($)": round(expense_per_unit, 2),
            "Occupancy Rate (%)": round(occupancy_rate * 100, 2),
            "Total Units": num_units,
            "Parking Income ($)": round(parking_income, 2),
            "Laundry Income ($)": round(laundry_income, 2),
            "Other Income ($)": round(total_other_income, 2),
            "Total Income ($)": round(total_income, 2),
            "Total Expenses ($)": round(total_expenses, 2),
            "Adjusted NOI ($)": round(noi, 2),
            "Gross Rent Multiplier": round(offer_price_value / (total_income * 12), 2) if total_income > 0 else 0
        }
        
    except Exception as e:
        logging.error(f"Error calculating metrics: {str(e)}")
        raise ValueError(f"Error calculating metrics: {str(e)}")
