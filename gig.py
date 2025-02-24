import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
import math
from datetime import datetime, timedelta

# --- Code 1 functions ---
def calculate_total_inventory_cost1(q, demand_rate, ordering_cost, holding_cost_per_unit):
    if q <= 0:
        return float('inf')  # Return infinity for non-positive q
    return (demand_rate / q) * ordering_cost + (q / 2) * holding_cost_per_unit

def optimize_eoq1(demand_rate, ordering_cost, holding_cost_per_unit):
    if demand_rate <= 0 or ordering_cost <= 0 or holding_cost_per_unit <= 0:
        return 0
    result = minimize_scalar(calculate_total_inventory_cost1, bounds=(0.001, demand_rate*10), args=(demand_rate, ordering_cost, holding_cost_per_unit), method='bounded')
    return result.x

def analyze_inventory1(df, ordering_cost_input, holding_cost_percentage):
    df['holding_cost_per_unit'] = df['avg_daily_demand'] * holding_cost_percentage
    df['Ordering Cost per Unit'] = ordering_cost_input
    df['annual_demand'] = df['avg_daily_demand'] * 365
    return df

# --- Code 2 functions ---
def analyze_inventory2(df, ordering_cost, holding_cost_percentage):
    try:
        # 1. Print Column Names for Debugging (Essential!)
        st.write("DataFrame Columns:", df.columns)

        # 2. Correct Column Name Matching (Based on your revised data)
        demand_col = 'avg daily demand (past 30 days)'  # Exact name from your data
        price_col = 'Price'        # Exact name from your data
        name_col = 'Name'          # Exact name from your data

        # Check if columns exist (more robust)
        if demand_col not in df.columns:
            st.error(f"Column '{demand_col}' not found in DataFrame.")
            return pd.DataFrame()  # Return empty DataFrame
        if price_col not in df.columns:
            st.error(f"Column '{price_col}' not found in DataFrame.")
            return pd.DataFrame()
        if name_col not in df.columns:
            st.error(f"Column '{name_col}' not found in DataFrame.")
            return pd.DataFrame()


        # 3. Data Cleaning and Conversion
        df['Price'] = df[price_col].astype(str).str.replace(r'[$,]', '', regex=True).str.replace(',', '').astype(float)
        df[demand_col] = df[demand_col].astype(str).str.replace(',', '').astype(float) # Clean demand column
        df['annual_demand'] = df[demand_col] * 365
        df['holding_cost_per_unit'] = df['Price'] * holding_cost_percentage
        df['eoq'] = np.sqrt((2 * df['annual_demand'] * ordering_cost) / df['holding_cost_per_unit'])
        df['total_inventory_cost_classic'] = (df['eoq'] / 2) * df['holding_cost_per_unit'] + (df['annual_demand'] / df['eoq']) * ordering_cost
        return df

    except Exception as e:
        st.error(f"Error in analyze_inventory2: {e}")
        return pd.DataFrame()  # Return empty DataFrame on error


# --- Code 3 functions ---
def calculate_eoq3(demand, ordering_cost, holding_cost):
    return math.sqrt((2 * demand * ordering_cost) / holding_cost)

def get_next_reorder_date(stocking_date, demand_per_day, eoq, lead_time):
    days_to_deplete_stock = eoq / demand_per_day
    reorder_date = stocking_date + timedelta(days=round(days_to_deplete_stock - lead_time))
    return reorder_date

def calculate_reorder_point(daily_demand, lead_time):
    return daily_demand * lead_time * 0.8  # Added 0.8 as a safety stock factor


# --- Code 4 functions ---
def calculate_eoq4(demand_rate, ordering_cost, holding_cost_per_unit):
    return np.sqrt((2 * demand_rate * ordering_cost) / holding_cost_per_unit)

def predict_sales_during_promotion(avg_daily_demand, sales_uplift, promo_duration):
    return avg_daily_demand * (1 + sales_uplift) * promo_duration

# --- Streamlit app ---
st.title("Inventory Management Tool")

code_selection = st.selectbox("Select Code to Run:", ["Basic inventory analysis with EOQ optimization", "Advanced inventory analysis, cost breakdowns, and visualizations", "Reorder point calculation and reorder date estimation", "EOQ calculations considering sales promotions"])

if code_selection == "Basic inventory analysis with EOQ optimization":
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            ordering_cost = st.number_input("Ordering Cost", value=10)
            holding_cost_percentage = st.number_input("Holding Cost Percentage", value=0.2)

            df.columns = df.columns.str.strip()
            df.rename(columns={'Name ': 'Name', 'avg daily demand (past 30 days)': 'avg_daily_demand'}, inplace=True)
            df['avg_daily_demand'] = pd.to_numeric(df['avg_daily_demand'], errors='coerce')
            df.dropna(subset=['avg_daily_demand'], inplace=True)

            if st.button("Analyze Inventory"):
                df = analyze_inventory1(df, ordering_cost, holding_cost_percentage)
                st.write(df)
                st.write(df.describe())

        except Exception as e:
            st.error(f"An error occurred: {e}")

if code_selection == "Advanced inventory analysis, cost breakdowns, and visualizations":
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            ordering_cost = st.number_input("Ordering Cost", value=10)
            holding_cost_percentage = st.number_input("Holding Cost Percentage", value=0.2)

            df.columns = df.columns.str.strip()
            demand_col = next((col for col in df.columns if 'avg daily' in col.lower()), None)
            price_col = next((col for col in df.columns if 'price' in col.lower()), None)
            name_col = next((col for col in df.columns if 'name' in col.lower()), None)

            if demand_col and price_col:
                if name_col:
                    product_names = df[name_col].unique().tolist()
                    selected_product = st.selectbox("Select a product for analysis:", product_names)
                else:
                    st.warning("No 'name' column found. Using the first product for analysis.")
                    selected_product = df.iloc[0, 0]
                    df['name'] = df.iloc[:, 0]

                if st.button("Analyze Inventory"):
                    df = analyze_inventory2(df, ordering_cost, holding_cost_percentage)

                    if not df.empty:
                        st.write(df)
                        st.write(df[['total_inventory_cost_classic']].describe())

                        example_product = df[df['Name'] == selected_product].iloc[0]

                        annual_demand_example = example_product['annual_demand']
                        ordering_cost_example = ordering_cost
                        holding_cost_per_unit_example = example_product['holding_cost_per_unit']

                        # --- EOQ Model Graph ---
                        max_reorder_quantity = max(100, int(annual_demand_example * 2))  # Set a reasonable upper bound
                        reorder_quantities = np.linspace(1, max_reorder_quantity, 500)

                        holding_costs = (reorder_quantities / 2) * holding_cost_per_unit_example
                        ordering_costs = (annual_demand_example / reorder_quantities) * ordering_cost_example
                        total_costs = holding_costs + ordering_costs

                        eoq_example = np.sqrt((2 * annual_demand_example * ordering_cost_example) / holding_cost_per_unit_example)
                        eoq_cost_example = (eoq_example / 2) * holding_cost_per_unit_example + (annual_demand_example / eoq_example) * ordering_cost_example

                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.plot(reorder_quantities, holding_costs, label='Holding Costs', color='orange')
                        ax.plot(reorder_quantities, ordering_costs, label='Ordering Costs', color='blue')
                        ax.plot(reorder_quantities, total_costs, label='Total Costs', color='red')
                        ax.axvline(eoq_example, color='black', linestyle='--', label=f'EOQ: {eoq_example:.2f}')
                        ax.scatter(eoq_example, eoq_cost_example, color='yellow', edgecolor='black', zorder=5, label=f'EOQ Point: {eoq_cost_example:.2f}')
                        ax.set_title(f'EOQ Model for {example_product["Name"]}')
                        ax.set_xlabel('Reorder Quantity')
                        ax.set_ylabel('Annual Cost')
                        ax.legend()
                        ax.grid(True)
                        fig.tight_layout()
                        st.pyplot(fig)  # Display the plot

                    else:
                        st.warning("Data analysis returned an empty DataFrame. Check your input data and calculations.")

            else:
                st.error("Please provide a valid CSV with 'avg daily der' and 'Price' data.")

        except Exception as e:
            st.error(f"An error occurred: {e}")

    
 # Part 5: Streamlit App (Continuation and Code 3 & 4)

elif code_selection == "Reorder point calculation and reorder date estimation":
    st.header("Code 3: Reorder Point and Date Calculation")
    holding_cost = st.number_input("Holding Cost (per unit per year)", value=0.1)
    ordering_cost = st.number_input("Ordering Cost (per order)", value=10)
    stocking_date_str = st.date_input("Stocking Date")
    demand_per_day = st.number_input("Daily Demand (units per day)", value=10)
    lead_time = st.number_input("Lead Time (in days)", value=7)

    if st.button("Calculate Reorder Point and Date"):
        try:
            stocking_date = datetime.combine(stocking_date_str, datetime.min.time())
            yearly_demand = demand_per_day * 365
            eoq = calculate_eoq3(yearly_demand, ordering_cost, holding_cost)
            reorder_point = calculate_reorder_point(demand_per_day, lead_time)
            next_reorder_date = get_next_reorder_date(stocking_date, demand_per_day, eoq, lead_time)

            st.write(f"Economic Order Quantity (EOQ): {eoq:.2f} units")
            st.write(f"Reorder Point: {reorder_point:.2f} units")
            st.write(f"Next Reordering Date: {next_reorder_date.strftime('%Y-%m-%d')}")
        except Exception as e:
            st.error(f"An error occurred: {e}")

elif code_selection == "EOQ calculations considering sales promotions":
    st.header("Code 4: EOQ During Promotion")
    avg_daily_demand = st.number_input("Enter the average daily demand (units): ", value=100)
    sales_uplift = st.number_input("Enter the sales uplift during promotion (as a decimal, e.g., 0.5 for 50%): ", value=0.5)
    promo_duration = st.number_input("Enter the promotion duration (days): ", value=30)
    ordering_cost = st.number_input("Enter the ordering cost ($ per order): ", value=10)
    holding_cost_percentage = st.number_input("Enter the holding cost as a percentage of item price (e.g., 0.1 for 10%): ", value=0.1)
    price_per_unit = st.number_input("Enter the price per unit ($): ", value=20)

    if st.button("Calculate EOQ During Promotion"):
        try:
            total_sales_demand = predict_sales_during_promotion(avg_daily_demand, sales_uplift, promo_duration)
            holding_cost_per_unit = price_per_unit * holding_cost_percentage
            eoq = calculate_eoq4(total_sales_demand, ordering_cost, holding_cost_per_unit)

            reorder_quantities = np.linspace(1, total_sales_demand, 500)
            holding_costs = (reorder_quantities / 2) * holding_cost_per_unit
            ordering_costs = (total_sales_demand / reorder_quantities) * ordering_cost
            total_costs = holding_costs + ordering_costs
            eoq_cost = (eoq / 2) * holding_cost_per_unit + (total_sales_demand / eoq) * ordering_cost

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(reorder_quantities, holding_costs, label='Holding Costs', color='orange')
            ax.plot(reorder_quantities, ordering_costs, label='Ordering Costs', color='blue')
            ax.plot(reorder_quantities, total_costs, label='Total Costs', color='red')
            ax.axvline(eoq, color='green', linestyle='--', label=f'EOQ: {eoq:.2f}')
            ax.scatter(eoq, eoq_cost, color='purple', label=f'EOQ Cost: {eoq_cost:.2f}', zorder=5)
            ax.set_title('EOQ Model During Sales Period')
            ax.set_xlabel('Reorder Quantity (Units)')
            ax.set_ylabel('Costs ($)')
            ax.legend()
            ax.grid(True)
            fig.tight_layout()
            st.pyplot(fig)

            st.write(f"Predicted Total Sales Demand: {total_sales_demand} units")
            st.write(f"EOQ During Sales: {eoq:.2f} units")
        except Exception as e:
            st.error(f"An error occurred: {e}")               
