# Methodology & Technical Documentation

This document explains the technical methods, algorithms, and statistical approaches used in the stock price prediction tool.

##  Table of Contents

1. [System Overview](#system-overview)
2. [Data Collection](#data-collection)
3. [Technical Indicators](#technical-indicators)
4. [Outlier Detection](#outlier-detection)
5. [Prophet Forecasting Model](#prophet-forecasting-model)
6. [Hyperparameter Optimization](#hyperparameter-optimization)
7. [Prediction Process](#prediction-process)
8. [Limitations & Assumptions](#limitations--assumptions)
9. [Future Improvements](#future-improvements)

---

##  System Overview

### Pipeline Architecture

```
User Input (Stock Symbol)
    ↓
Yahoo Finance API (yfinance)
    ↓
Data Preprocessing & Feature Engineering
    ↓
Outlier Detection (Elliptic Envelope)
    ↓
Data Scaling (Min-Max Normalization)
    ↓
Model Training (Prophet with Cross-Validation)
    ↓
Price Prediction (365 days)
    ↓
Visualization (Matplotlib)
```

### Core Technologies

- **yfinance**: Yahoo Finance API wrapper for data retrieval
- **Prophet**: Facebook's time series forecasting framework
- **scikit-learn**: Machine learning utilities (scaling, outlier detection)
- **pandas/numpy**: Data manipulation and numerical computing
- **matplotlib**: Visualization and plotting

---

##  Data Collection

### Yahoo Finance API

```python
stock_data = yf.download(symbol, period="max")
```

**Retrieved Data**:
- **Date**: Trading dates
- **Open/High/Low/Close**: Daily prices
- **Volume**: Shares traded
- **Adjusted Close**: Accounts for splits and dividends

### Data Preprocessing

```python
df = stock_data.reset_index()
df = df[['Date', 'Open', 'Volume']]
df.columns = ['ds', 'y', 'Volume']  # Prophet naming convention
```

**Why Open Price?**
- Represents market sentiment at day start
- Useful for intraday strategies
- Can be changed to Close if preferred

**Missing Data**: Currently uses `dropna()` to remove incomplete records. Future versions could use forward-fill for better data retention.

---

## Technical Indicators

### 1. Moving Averages (MA)

**Formula**:
```
MA(n) = (P₁ + P₂ + ... + Pₙ) / n
```

**Implementation**:
```python
df['MA_20'] = df['y'].rolling(window=20).mean()  # Short-term
df['MA_50'] = df['y'].rolling(window=50).mean()  # Medium-term
```

**Interpretation**:
- **Golden Cross** (MA_20 > MA_50): Bullish signal
- **Death Cross** (MA_20 < MA_50): Bearish signal
- Price above both MAs indicates strong uptrend

### 2. Relative Strength Index (RSI)

**Formula**:
```
RSI = 100 - (100 / (1 + RS))
where RS = Average Gain / Average Loss
```

**Implementation**:
```python
def calculate_rsi(data, window=14):
    delta = data['y'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi
```

**Interpretation**:
- **RSI > 70**: Overbought (potential reversal down)
- **RSI < 30**: Oversold (potential reversal up)
- **RSI ≈ 50**: Neutral momentum

**Current Use**: Feature for outlier detection and model input

---

## 🔍 Outlier Detection

### Elliptic Envelope Algorithm

**Purpose**: Remove anomalous data points that could distort predictions

**Method**: Assumes data follows Gaussian distribution and fits a robust covariance estimate. Points outside the fitted ellipse are flagged as outliers.

**Implementation**:
```python
outlier_detector = EllipticEnvelope(
    contamination=0.03,      # Expect 3% outliers
    support_fraction=0.8      # Use 80% for covariance estimation
)
outlier_mask = outlier_detector.fit_predict(df[['y', 'Volume', 'RSI']])
df = df[outlier_mask == 1]  # Keep only inliers
```

**Parameters**:
- **contamination=0.03**: Assumes 3% of data are outliers (conservative for financial data)
- **support_fraction=0.8**: Uses 80% of data for robust estimation

**Features Analyzed**:
- Price (y)
- Volume
- RSI

**Why This Matters**: Removes extreme events (e.g., flash crashes, data errors) that could skew the model

### Data Scaling

**Min-Max Normalization**:
```python
scaler = MinMaxScaler()
df[['y', 'Volume', 'RSI']] = scaler.fit_transform(df[['y', 'Volume', 'RSI']])
```

**Formula**: `X_scaled = (X - X_min) / (X_max - X_min)`

**Purpose**: Normalizes features to [0,1] range for equal importance and better model convergence

---

##  Prophet Forecasting Model

### What is Prophet?

Prophet is an additive regression model developed by Facebook for time series forecasting. It's particularly good at handling:
- Missing data
- Trend changes
- Seasonality patterns
- Holiday effects

### Model Components

**Mathematical Formula**:
```
y(t) = g(t) + s(t) + h(t) + εₜ

where:
  g(t) = trend (piecewise linear growth)
  s(t) = seasonality (Fourier series)
  h(t) = holiday effects
  εₜ = error term
```

### 1. Trend Component

- Captures long-term price direction
- Automatically detects changepoints (trend shifts)
- Piecewise linear model allows flexibility

### 2. Seasonality Components

**Yearly Seasonality** (`yearly_seasonality=True`):
- Captures annual patterns (e.g., January Effect, December Rally)
- Uses 10 Fourier terms by default
- Period = 365.25 days

**Daily Seasonality** (`daily_seasonality=True`):
- Captures within-week patterns (e.g., Monday Effect)
- Uses 4 Fourier terms
- Period = 7 days

### 3. Holiday Effects

```python
model.add_country_holidays(country_name='US')  # NYSE holidays

# Custom market holidays
market_holidays = pd.DataFrame({
    'holiday': 'market_holiday',
    'ds': pd.to_datetime(['2022-01-01', '2022-12-25', ...]),
    'lower_window': 0,
    'upper_window': 1
})
```

**US Market Holidays**: New Year's, MLK Day, Presidents' Day, Good Friday, Memorial Day, Independence Day, Labor Day, Thanksgiving, Christmas

---

## Hyperparameter Optimization

### Cross-Validation Strategy

**Time Series Cross-Validation**:
```python
df_cv = cross_validation(
    model, 
    initial='730 days',    # 2 years training
    period='90 days',      # Step size between cutoffs
    horizon='180 days'     # 6 months prediction horizon
)
```

**Process**:
1. Train on first 730 days
2. Predict next 180 days
3. Move forward 90 days
4. Repeat until data exhausted

**Why This Method?**: Preserves temporal ordering (critical for time series) and tests model on multiple time periods

### Parameter Grid

```python
param_grid = {
    'changepoint_prior_scale': [0.01, 0.1],
    'seasonality_prior_scale': [1.0, 10.0],
    'holidays_prior_scale': [1.0, 10.0]
}
```

**Total Combinations**: 2 × 2 × 2 = 8

**Parameter Meanings**:

1. **changepoint_prior_scale**: Controls trend flexibility
   - Lower (0.01): Conservative, smooth trends
   - Higher (0.1): Flexible, captures rapid changes

2. **seasonality_prior_scale**: Controls seasonality strength
   - Lower (1.0): Weaker seasonal effects
   - Higher (10.0): Stronger seasonal patterns

3. **holidays_prior_scale**: Controls holiday impact
   - Lower (1.0): Minimal holiday effects
   - Higher (10.0): Strong holiday influence

### Selection Criterion

**RMSE (Root Mean Square Error)**:
```
RMSE = √(Σ(yᵢ - ŷᵢ)² / n)
```

The parameter combination with the lowest RMSE is selected for final predictions.

**Computational Note**: Uses subset of data (2020-2022) for tuning to reduce computation time while maintaining representativeness.

---

## 📊 Prediction Process

### Step 1: Generate Future Dates

```python
future_dates = pd.date_range(
    start=datetime.now(), 
    end=datetime.now() + timedelta(days=365), 
    freq='D'
)
```

**Horizon**: 365 days (1 year) into the future

### Step 2: Make Predictions

```python
forecast = best_model.predict(future_df)
```

**Forecast Output**:
- `yhat`: Point estimate (predicted price)
- `yhat_lower`: Lower bound of 80% confidence interval
- `yhat_upper`: Upper bound of 80% confidence interval
- Component breakdowns (trend, seasonality, holidays)

### Step 3: Inverse Transform

```python
forecast[['yhat', 'yhat_lower', 'yhat_upper']] = scaler.inverse_transform(
    forecast[['yhat', 'yhat_lower', 'yhat_upper']]
)
```

Converts scaled [0,1] values back to original price scale

### Step 4: Price Adjustment

```python
latest_price = data['Close'].iloc[-1]
forecast['yhat'] = latest_price * (forecast['yhat'] / forecast['yhat'].iloc[0])
forecast['yhat_lower'] = latest_price * (forecast['yhat_lower'] / forecast['yhat_lower'].iloc[0])
forecast['yhat_upper'] = latest_price * (forecast['yhat_upper'] / forecast['yhat_upper'].iloc[0])
```

**Purpose**: Anchors predictions to current price, ensuring continuity between historical and forecasted prices

**Formula**: `Adjusted_Price(t) = Current_Price × (Predicted_Price(t) / Predicted_Price(0))`

### Visualization

**Main Plot Components**:
- Historical prices (last 3 months)
- Predicted prices (next 12 months)
- 80% confidence interval (shaded area)
- Current date marker (vertical line)
- Latest price reference (horizontal line)

**Component Plot**:
- Trend decomposition
- Yearly seasonality pattern
- Daily seasonality pattern
- Holiday effects

---

## Limitations & Assumptions

### Model Assumptions

1. **Historical Patterns Repeat**: Assumes past price behavior predicts future behavior
   - Reality: Markets evolve, unprecedented events occur

2. **Stationarity**: Assumes statistical properties remain constant
   - Reality: Regime changes (bull/bear markets) alter dynamics

3. **Linear Additivity**: Components add linearly
   - Reality: Complex non-linear interactions exist

4. **No External Factors**: Doesn't incorporate:
   - Earnings reports
   - News sentiment
   - Economic indicators (interest rates, GDP, inflation)
   - Competitive landscape
   - Management changes

### Data Limitations

1. **Historical Data Only**: No forward-looking information
2. **Yahoo Finance Dependency**: Data quality depends on provider
3. **Missing Fundamental Analysis**: P/E ratios, revenue, etc. not considered
4. **No Sentiment Analysis**: Social media, news not included

### Prediction Limitations

1. **Extrapolation Risk**: 365-day forecasts are highly uncertain
2. **Black Swan Events**: Cannot predict pandemics, wars, crashes
3. **Overfitting Potential**: Complex model may fit noise
4. **Uncertainty Increases**: Confidence intervals widen over time

### Technical Limitations

1. **Computational Cost**: Hyperparameter tuning is slow (3-6 minutes)
2. **Internet Required**: Real-time data fetching needed
3. **API Rate Limits**: Yahoo Finance may throttle requests
4. **Memory Constraints**: Very long histories may exhaust RAM

### Important Disclaimer

**This tool is for educational purposes only and NOT financial advice.**
- Past performance does not guarantee future results
- Always consult a financial advisor before investing
- Never invest more than you can afford to lose

---

##  Future Improvements

### Model Enhancements

1. **Additional Models**: LSTM, ARIMA, XGBoost for comparison/ensemble
2. **Sentiment Analysis**: Integrate news and social media sentiment
3. **Fundamental Data**: Add P/E ratios, earnings, revenue metrics
4. **Multi-Stock Analysis**: Portfolio optimization and correlation analysis
5. **Regime Detection**: Identify and adapt to bull/bear markets

### Feature Engineering

1. **More Technical Indicators**: MACD, Bollinger Bands, Stochastic Oscillator
2. **Volume Analysis**: OBV, VWAP
3. **Price Transforms**: Log returns, percentage changes
4. **External Regressors**: Economic indicators, sector indices

### Performance Optimizations

1. **Caching**: Save downloaded data and trained models
2. **Parallel Processing**: Analyze multiple stocks simultaneously
3. **Incremental Updates**: Update models with new data rather than retraining

### User Experience

1. **Web Interface**: Streamlit or Dash dashboard
2. **Interactive Plots**: Plotly for zoom, pan, hover details
3. **Export Features**: Save predictions to CSV/Excel
4. **Real-time Alerts**: Notify when price targets are reached
5. **Backtesting**: Test strategies on historical data

### Code Quality

1. **Unit Tests**: Automated testing for reliability
2. **Type Hints**: Better code documentation and IDE support
3. **Modular Design**: Separate concerns into classes/modules
4. **Logging**: Track errors and performance metrics

---

##  Key References

**Prophet Documentation**: https://facebook.github.io/prophet/docs/quick_start.html

**Technical Papers**:
- Taylor & Letham (2018). "Forecasting at Scale." *The American Statistician*
- Fama (1970). "Efficient Capital Markets" *Journal of Finance*

**Financial Concepts**:
- Investopedia: Technical Analysis, RSI, Moving Averages
- QuantConnect: Algorithmic trading education

**Python Libraries**:
- yfinance: https://github.com/ranaroussi/yfinance
- scikit-learn: https://scikit-learn.org/
- pandas: https://pandas.pydata.org/

---

## Support

**Questions?** Open an issue on GitHub or email [leon.angelf15@gmail.com]


---

*Last Updated: 2024 | Version: 1.0*
