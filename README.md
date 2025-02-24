# Inventory Management Tool

This Streamlit application provides a suite of tools for analyzing and optimizing inventory management in e-commerce. It implements several inventory models and calculations, allowing users to upload their data and explore different inventory strategies.

## Features

The tool is divided into four main sections, each implementing a different inventory management approach:

**Code 1: Basic Inventory Analysis**

*   Calculates basic inventory metrics based on provided data, including annual demand.
*   Allows users to input ordering cost and holding cost percentage.
*   Provides a table of results and descriptive statistics.

**Code 2: Enhanced EOQ Analysis with Graphing**

*   Calculates the Economic Order Quantity (EOQ) using an optimized approach.
*   Handles data cleaning and type conversion for common data issues.
*   Provides a detailed table of results, including total inventory costs.
*   Generates an interactive graph visualizing the EOQ model, showing holding costs, ordering costs, total costs, and the EOQ point.
*   Allows users to select a specific product for in-depth analysis and graph generation.

**Code 3: Reorder Point and Date Calculation**

*   Calculates the reorder point and the next reorder date based on user inputs.
*   Considers lead time and daily demand to determine when to reorder inventory.
*   Provides clear output of EOQ, reorder point, and next reorder date.

**Code 4: EOQ During Promotion**

*   Calculates the EOQ during a promotional period, considering sales uplift and promotion duration.
*   Predicts total sales demand during the promotion.
*   Generates an interactive graph showing the EOQ model during the sales period.
*   Provides output of predicted total sales demand and EOQ during sales.

## Getting Started

1.  **Clone the repository (optional):** If you want to modify the code, clone the repository to your local machine.

2.  **Install dependencies:** Make sure you have the required Python libraries installed. You can install them using pip:

    ```bash
    pip install streamlit pandas numpy scipy matplotlib
    ```

3.  **Run the app:** Navigate to the directory containing the script and run the Streamlit app:

    ```bash
    streamlit run your_script_name.py  # Replace your_script_name.py with the actual name
    ```

4.  **Upload your data:** The app will open in your web browser. Upload your CSV file containing your inventory data. Ensure your CSV has the necessary columns (e.g., 'Name', 'Price', 'avg daily demand (past 30 days)').  The column names can be flexible, and the code will attempt to find them based on keywords (e.g., "avg daily", "price", "name").

5.  **Input parameters:** Enter the required parameters, such as ordering cost and holding cost percentage.

6.  **Analyze:** Click the "Analyze Inventory" or other relevant buttons to perform the calculations and view the results.

## Data Format

The CSV file should contain data in a tabular format.  The code is designed to be somewhat flexible with column names, but it's best to have columns that at least contain the following information:

*   **Product Name:** A column that identifies each product (e.g., "Name", "Product Name").
*   **Price:** A column containing the price of each product (e.g., "Price").  Currency symbols and commas should be handled, but it's best to have clean numerical data.
*   **Average Daily Demand:** A column with the average daily demand for each product (e.g., "avg daily demand (past 30 days)", "Daily Demand").

## Examples

The app provides examples of how to use each of the four code sections.  Refer to the app interface for specific instructions and parameter descriptions for each section.

## Contributing

Contributions are welcome! If you find any bugs or have suggestions for improvements, please open an issue or submit a pull request.

## License

[Add your license here]
