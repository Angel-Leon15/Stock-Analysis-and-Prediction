# Stock Analysis and Prediction Tool

This Python project utilizes various libraries to fetch, analyze, and predict stock prices using historical data from Yahoo Finance. It is designed to provide insights into stock trends, basic properties, and option chains, as well as forecast future prices using Facebook's Prophet library for time series forecasting.

## Features

- **Stock Data Retrieval**: Fetch historical stock data.
- **Stock Information Display**: Show detailed information about the stock such as bid price, open price, previous close, etc.
- **Option Chain Retrieval**: Get the option chain for a stock for a specific expiration date.
- **Basic Stock Properties Display**: Show basic properties of the stock like the current price and volume.
- **Stock Price Prediction**: Analyze trends and predict future stock prices using advanced statistical models.

---

## 🛠️ Libraries Used
- [yfinance](https://pypi.org/project/yfinance/) – Fetch stock data from Yahoo Finance.  
- [prophet](https://facebook.github.io/prophet/) – Time series forecasting.  
- [pandas](https://pandas.pydata.org/) – Data manipulation and analysis.  
- [numpy](https://numpy.org/) – Numerical computations.  
- [matplotlib](https://matplotlib.org/) – Visualization and plotting.  
- [scikit-learn](https://scikit-learn.org/) – Preprocessing and machine learning utilities.  

---



## Setup and Installation
## Setup and Installation

1. **Download the project:**
   - Download the project as a ZIP file from [this link](https://github.com/Angel-Leon15/Stock-Analysis-and-Prediction/blob/main/ML_Stock_Project.zip).

2. **Extract the ZIP file:**
   - Once downloaded, extract the contents of the ZIP file to a location of your choice on your computer.

3. **Navigate to the project directory:**
   - Open a terminal or command prompt and change directory to the location where you extracted the project.

4. **Install required libraries:**
   - Run the following command to install the necessary Python libraries:
     ```bash
     pip install yfinance prophet pandas numpy matplotlib scikit-learn
     ```

Now, you're ready to use the project by running the script.



## Usage
Run the script from the command line:
   ```bash
   python ML_Stock_Project.py
   ```
Follow the interactive prompts in the command line interface to choose different options like viewing stock data, retrieving stock information, or predicting future prices.

## Contributing

Contributions are welcome! For major changes, please open an issue first to discuss what you would like to change.

## License

Distributed under the MIT License. See `LICENSE` for more information.
