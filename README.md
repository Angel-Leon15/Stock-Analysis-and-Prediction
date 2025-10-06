# ML Stock Price Prediction & Analysis Tool

A comprehensive machine learning tool for stock market analysis and price prediction using Facebook's Prophet algorithm, technical indicators, and advanced statistical methods.

## Project Overview

This project provides an interactive command-line application that fetches real-time stock data, performs technical analysis, detects outliers, and generates future price predictions using time series forecasting. The tool leverages Prophet's powerful forecasting capabilities combined with traditional technical indicators to provide insights into stock price movements.

##  Key Features

- **Real-Time Data Fetching**: Pulls historical and current stock data from Yahoo Finance API
- **Technical Analysis**: Calculates moving averages (MA20, MA50) and Relative Strength Index (RSI)
- **Outlier Detection**: Uses Elliptic Envelope algorithm to identify and remove anomalous data points
- **Time Series Forecasting**: Prophet-based predictions with uncertainty intervals
- **Hyperparameter Optimization**: Automated cross-validation to find optimal model parameters
- **Seasonality Analysis**: Captures daily and yearly patterns in stock prices
- **Market Holidays**: Incorporates US market holidays for improved accuracy
- **Interactive Visualizations**: Displays historical prices, predictions, and forecast components
- **Option Chain Analysis**: Retrieves and displays options data for specific expiration dates

## Use Cases

- **Traders & Investors**: Identify potential price trends and make informed decisions
- **Data Scientists**: Study time series forecasting techniques in financial markets
- **Students**: Learn about machine learning applications in finance
- **Analysts**: Generate technical reports with statistical backing

## 📁 Project Structure

```
.
├── ML_Stock_Project.py          # Main application file
├── README.md                     # Project documentation
├── CONTRIBUTING.md               # Contribution guidelines
├── METHODOLOGY.md                # Technical documentation
└── requirements.txt              # Python dependencies
```

## Prerequisites

### System Requirements

- Python 3.8 or higher
- Internet connection (for fetching stock data)
- Minimum 4GB RAM
- Display capable of showing matplotlib plots

### Required Python Packages

```bash
pip install yfinance prophet pandas numpy matplotlib scikit-learn
```

Or use the requirements file:

```bash
pip install -r requirements.txt
```

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `yfinance` | ≥0.2.0 | Yahoo Finance API wrapper |
| `prophet` | ≥1.1 | Time series forecasting |
| `pandas` | ≥1.3.0 | Data manipulation |
| `numpy` | ≥1.21.0 | Numerical computing |
| `matplotlib` | ≥3.4.0 | Visualization |
| `scikit-learn` | ≥1.0.0 | Machine learning utilities |

##  Getting Started

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/ml-stock-prediction.git
   cd ml-stock-prediction
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python ML_Stock_Project.py
   ```

### Quick Start Guide

1. **Launch the program**
   ```bash
   python ML_Stock_Project.py
   ```

2. **Enter a stock symbol** (e.g., AAPL, GOOGL, TSLA)

3. **Choose from available options**:
   - `1`: Display basic properties (current price, volume)
   - `2`: Display detailed stock information
   - `3`: Retrieve option chain data
   - `4`: Change stock symbol
   - `5`: Analyze trends and predict future prices
   - `6`: Exit program

### Example Session

```
Enter stock symbol: AAPL

Options:
1. Display basic properties
2. Display stock information
3. Retrieve option chain
4. Change stock symbol
5. Analyze trends and predict future prices
6. Exit

Enter your choice (1-6): 5

[Generates prediction plot and component analysis]
```

##  Features in Detail

### 1. Real-Time Stock Data

Fetches comprehensive historical data including:
- Open, High, Low, Close prices
- Trading volume
- Adjusted close prices
- Complete price history

### 2. Technical Indicators

**Moving Averages**:
- MA20: 20-day moving average (short-term trend)
- MA50: 50-day moving average (medium-term trend)

**Relative Strength Index (RSI)**:
- 14-period RSI calculation
- Identifies overbought (>70) and oversold (<30) conditions

### 3. Outlier Detection

- **Algorithm**: Elliptic Envelope (robust covariance estimation)
- **Contamination Rate**: 3% of data treated as outliers
- **Features Used**: Price, Volume, RSI
- **Purpose**: Remove anomalous data points that could skew predictions

### 4. Price Prediction

**Forecast Horizon**: 365 days (1 year) into the future

**Model Features**:
- Daily and yearly seasonality
- US market holiday effects
- Uncertainty intervals (confidence bands)
- Hyperparameter-optimized Prophet model

**Visualization**:
- Historical prices (last 3 months)
- Predicted prices (next 12 months)
- Confidence intervals (uncertainty bands)
- Component breakdown (trend, seasonality, holidays)

### 5. Option Chain Analysis

Retrieve calls and puts data for any available expiration date:
- Strike prices
- Bid/Ask prices
- Implied volatility
- Open interest

## 📈 Model Performance

The prediction model uses cross-validation to optimize hyperparameters:

- **Initial Training Period**: 730 days (2 years)
- **Validation Period**: 90 days
- **Test Horizon**: 180 days
- **Performance Metric**: RMSE (Root Mean Square Error)

**Hyperparameters Tuned**:
- `changepoint_prior_scale`: Controls trend flexibility
- `seasonality_prior_scale`: Controls seasonality strength
- `holidays_prior_scale`: Controls holiday effect strength

## ⚠️ Important Disclaimers

### Investment Warning

**This tool is for educational and informational purposes only.**

- ❌ NOT financial advice
- ❌ NOT a guarantee of future performance
- ❌ Past performance does not indicate future results
- ❌ Always consult with a qualified financial advisor before making investment decisions

### Model Limitations

- Predictions are based on historical patterns
- Cannot predict unprecedented market events (black swans)
- Market conditions change; models may become outdated
- External factors (news, policy changes) not incorporated
- Higher uncertainty for longer prediction horizons

## Troubleshooting

### Common Issues

**Issue**: "No module named 'prophet'"
```bash
# Solution: Install prophet with correct command
pip install prophet
# or
conda install -c conda-forge prophet
```

**Issue**: "Error fetching data"
- Check internet connection
- Verify stock symbol is valid
- Yahoo Finance API may be temporarily unavailable

**Issue**: Matplotlib plots not displaying
```python
# Add this to your environment
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg'
```

**Issue**: Long execution time for prediction
- Normal for first run (model training takes time)
- Subsequent predictions are faster
- Consider reducing date range for hyperparameter tuning

##  Future Enhancements

- [ ] GUI interface using Tkinter or Streamlit
- [ ] Portfolio analysis (multiple stocks)
- [ ] Sentiment analysis from news/social media
- [ ] Real-time prediction updates
- [ ] Export predictions to CSV/Excel
- [ ] Backtesting framework for strategy evaluation
- [ ] Integration with additional data sources
- [ ] LSTM/GRU neural network models
- [ ] Ensemble predictions (multiple models)
- [ ] Risk metrics (Value at Risk, Sharpe Ratio)

##  Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- [Facebook Prophet](https://facebook.github.io/prophet/) for the forecasting framework
- [yfinance](https://github.com/ranaroussi/yfinance) for Yahoo Finance API access
- The open-source community for excellent Python libraries

## 📚 Resources

- [Prophet Documentation](https://facebook.github.io/prophet/docs/quick_start.html)
- [Technical Analysis Primer](https://www.investopedia.com/terms/t/technicalanalysis.asp)
- [Time Series Forecasting Guide](https://otexts.com/fpp3/)

---

**Disclaimer**: This software is provided "as is" without warranty of any kind. Use at your own risk. The authors are not responsible for any financial losses incurred from using this tool.
