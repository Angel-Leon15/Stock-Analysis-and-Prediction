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

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/stock-analysis-prediction-tool.git
cd stock-analysis-prediction-tool
```

Install Dependecies 
```bash
pip install -r requirements.txt
```

Run The Main Script 
```bash
python ML_Stock_Project.py
```

Follow the interactive prompts to:

- **View stock data**

- **Retrieve stock information**

- **Access option chains**

- **Predict future stock prices**

---

## Project Structure

```bash
Stock-Analysis-and-Prediction/
├── ML_Stock_Project.py      # Main script
├── requirements.txt         # Project dependencies
├── README.md                # Documentation
├── LICENSE                  # License (MIT)
├── .gitignore               # Git ignore rules
├── Methodology
```




## Contributing

Contributions are welcome! These are areas I'm looking to imporve and add: 
- Add more advanced machine learning models (LSTMs, XGBoost).  
- Include more financial indicators (MACD, RSI, Bollinger Bands).  
- Integrate sentiment analysis from news or social media.  
- Build a simple web dashboard for visualizati

## License

Distributed under the MIT License. See `LICENSE` for more information.
