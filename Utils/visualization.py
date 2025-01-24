import matplotlib.pyplot as plt
import streamlit as st
import logging
import numpy as np

def plot_metrics(metrics, chart_type="bar", save_path=None):
    """
    Plot financial metrics as a chart.
    
    Args:
        metrics: Dictionary of metrics (key-value pairs)
        chart_type: Type of chart ("bar", "pie", "line")
        save_path: Optional path to save the chart image
    """
    if not metrics or all(value == 0 for value in metrics.values()):
        st.warning("No meaningful data to plot.")
        return

    # Ensure metrics are numeric
    numeric_metrics = {}
    for k, v in metrics.items():
        try:
            numeric_metrics[k] = float(v)
        except (ValueError, TypeError):
            logging.warning(f"Skipping non-numeric value for {k}: {v}")
            continue

    if not numeric_metrics:
        st.error("No numeric values found in metrics. Please check your data.")
        return

    try:
        # Create the figure
        fig, ax = plt.subplots(figsize=(12, 7))

        if chart_type == "bar":
            # Bar chart
            bars = ax.bar(
                numeric_metrics.keys(), 
                numeric_metrics.values(), 
                color='skyblue', 
                edgecolor='black'
            )
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width()/2,
                    height,
                    f'${height:,.0f}',
                    ha='center',
                    va='bottom'
                )
            
            ax.set_title("Financial Metrics", fontsize=16, fontweight='bold')
            ax.set_ylabel("Value ($)", fontsize=12)
            ax.set_xlabel("Metrics", fontsize=12)
            plt.xticks(rotation=45, ha='right')
            ax.grid(axis='y', linestyle='--', alpha=0.7)

        elif chart_type == "pie":
            # Check if pie chart is suitable
            if len(numeric_metrics) > 8:
                st.warning("Too many metrics for a pie chart. Switching to bar chart.")
                plot_metrics(metrics, chart_type="bar", save_path=save_path)
                return
                
            # Pie chart
            plt.pie(
                numeric_metrics.values(),
                labels=numeric_metrics.keys(),
                autopct='$%1.0f',
                startangle=90,
                colors=plt.cm.Paired(np.linspace(0, 1, len(numeric_metrics)))
            )
            ax.set_title("Financial Metrics Distribution", fontsize=16, fontweight='bold')

        elif chart_type == "line":
            # Line chart
            ax.plot(
                list(numeric_metrics.keys()),
                list(numeric_metrics.values()),
                marker='o',
                linestyle='-',
                linewidth=2,
                color='skyblue'
            )
            
            # Add value labels
            for x, y in zip(numeric_metrics.keys(), numeric_metrics.values()):
                ax.text(x, y, f'${y:,.0f}', ha='center', va='bottom')
            
            ax.set_title("Financial Metrics Trend", fontsize=16, fontweight='bold')
            ax.set_ylabel("Value ($)", fontsize=12)
            ax.set_xlabel("Metrics", fontsize=12)
            plt.xticks(rotation=45, ha='right')
            ax.grid(True, linestyle='--', alpha=0.7)

        else:
            st.error(f"Unsupported chart type: {chart_type}")
            return

        # Adjust layout and display
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        # Display in Streamlit
        st.pyplot(fig)
        
        # Offer download if saved
        if save_path:
            with open(save_path, "rb") as f:
                st.download_button(
                    "Download Chart",
                    f,
                    file_name="financial_metrics.png",
                    mime="image/png"
                )

    except Exception as e:
        logging.error(f"Error plotting metrics: {str(e)}")
        st.error(f"Failed to create chart: {str(e)}")
    finally:
        plt.close()
