import yfinance as yf  # Used to fetch financial data from Yahoo Finance API
from prophet import Prophet  # Facebook's tool for time series forecasting
from prophet.diagnostics import cross_validation, performance_metrics  # Tools for model evaluation and validation
import pandas as pd  # Data manipulation and analysis library
import numpy as np  # Numerical computing library
import matplotlib.pyplot as plt  # Plotting library for creating static, interactive, and animated visualizations
from datetime import datetime, timedelta  # Standard library for handling dates and times
import matplotlib.dates as mdates  # Provides date-specific plotting tools for matplotlib
from sklearn.preprocessing import MinMaxScaler  # Feature scaling tool from scikit-learn
from sklearn.covariance import EllipticEnvelope  # Outlier detection model from scikit-learn
import itertools  # Standard library for creating iterators for efficient looping

def get_stock_data(symbol):
    try:
        stock = yf.Ticker(symbol)  # Create a Ticker object for the specified symbol
        data = stock.history(period="max")  # Fetch historical data for the stock
        return data
    except Exception as e:
        print(f"Error fetching data: {e}")  # Print error message if data fetching fails
        return None

def get_stock_info(symbol):
    try:
        stock = yf.Ticker(symbol)  # Create a Ticker object for the specified symbol
        info = stock.info  # Retrieve stock information such as company details, financials, etc.
        return info
    except Exception as e:
        print(f"Error fetching stock information: {e}")  # Print error message if information fetching fails
        return None

def get_option_chain(symbol, expiration_date):
    try:
        stock = yf.Ticker(symbol)  # Create a Ticker object for the specified symbol
        option_chain = stock.option_chain(expiration_date)  # Fetch the option chain for the given expiration date
        return option_chain
    except Exception as e:
        print(f"Error fetching option chain: {e}")  # Print error message if option chain fetching fails
        return None

def display_basic_properties(data):
    if data is not None:
        print("Basic Properties:")
        print("Current Price:", data['Close'].iloc[-1])  # Display the most recent closing price
        print("Volume:", data['Volume'].iloc[-1])  # Display the most recent trading volume
        # Add more properties as needed

def display_stock_info(info):
    if info is not None:
        print("Stock Information:")
        keys_to_display = ['bid', 'open', 'previousClose', 'dayHigh', 'dayLow', 'recommendationKey', 'mostRecentQuarter']
        for key in keys_to_display:
            if key in info:
                print(f"{key.capitalize()}: {info[key]}")  # Display each key's value if available
            else:
                print(f"{key.capitalize()}: N/A")  # Display N/A if key is not available
    else:
        print("Unable to retrieve stock information.")  # Print message if stock information is not available

# Analyze trends and predict future prices
def analyze_and_predict(symbol):
    # Retrieve stock data using yfinance
    stock_data = yf.download(symbol, period="max")

    # Get the latest stock price
    ticker_yahoo = yf.Ticker(symbol)
    data = ticker_yahoo.history()
    latest_price = data['Close'].iloc[-1]

    # Reset the index and rename the columns
    df = stock_data.reset_index()
    df = df[['Date', 'Open', 'Volume']]
    df.columns = ['ds', 'y', 'Volume']

    # Calculate moving averages
    df['MA_20'] = df['y'].rolling(window=20).mean()
    df['MA_50'] = df['y'].rolling(window=50).mean()

    # Calculate RSI
    def calculate_rsi(data, window=14):
        delta = data['y'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    df['RSI'] = calculate_rsi(df)

    # Handle missing data
    df.dropna(inplace=True)

    # Detect and remove outliers using Elliptic Envelope
    outlier_detector = EllipticEnvelope(contamination=0.03, support_fraction=0.8)
    outlier_mask = outlier_detector.fit_predict(df[['y', 'Volume', 'RSI']])
    df = df[outlier_mask == 1]

    # Scale the data using Min-Max Scaler
    scaler = MinMaxScaler()
    df[['y', 'Volume', 'RSI']] = scaler.fit_transform(df[['y', 'Volume', 'RSI']])

    # Create the Prophet model with seasonality and holidays
    model = Prophet(yearly_seasonality=True, daily_seasonality=True)
    model.add_country_holidays(country_name='US')

    # Add market-specific holidays (e.g., New Year's Day, Christmas)
    market_holidays = pd.DataFrame({
        'holiday': 'market_holiday',
        'ds': pd.to_datetime(['2022-01-01', '2022-12-25', '2023-01-01', '2023-12-25']),
        'lower_window': 0,
        'upper_window': 1
    })
    model.holidays = market_holidays

    # Perform cross-validation for hyperparameter tuning
    param_grid = {
        'changepoint_prior_scale': [0.01, 0.1],
        'seasonality_prior_scale': [1.0, 10.0],
        'holidays_prior_scale': [1.0, 10.0]
    }

    # Generate all possible combinations of hyperparameters
    param_combinations = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]

    # Use a subset of the data for hyperparameter tuning
    subset_data = df[(df['ds'] >= '2020-01-01') & (df['ds'] <= '2022-12-31')]

    # Perform cross-validation for each combination of hyperparameters
    cv_results = []
    for params in param_combinations:
        model = Prophet(**params)
        model.fit(subset_data)
        df_cv = cross_validation(model, initial='730 days', period='90 days', horizon='180 days')
        df_performance = performance_metrics(df_cv)
        cv_results.append({'params': params, 'performance': df_performance['rmse'].values[0]})

    # Select the best hyperparameters based on the performance metric (e.g., RMSE)
    best_params = min(cv_results, key=lambda x: x['performance'])['params']
    best_model = Prophet(**best_params)
    best_model.fit(df)

    # Set the start date for prediction to the current date
    prediction_start_date = datetime.now().strftime("%Y-%m-%d")
    prediction_end_date = (datetime.now() + timedelta(days=365)).strftime("%Y-%m-%d")

    # Generate future dates for prediction
    future_dates = pd.date_range(start=prediction_start_date, end=prediction_end_date, freq='D')
    future_df = pd.DataFrame({"ds": future_dates})

    # Make predictions with uncertainty intervals
    forecast = best_model.predict(future_df)

    # Inverse transform the scaled predicted prices
    forecast[['yhat', 'yhat_lower', 'yhat_upper']] = scaler.inverse_transform(forecast[['yhat', 'yhat_lower', 'yhat_upper']])

    # Set the initial predicted price to the latest price
    initial_price = latest_price

    # Scale the predicted prices based on the initial price
    forecast['yhat'] = initial_price * (forecast['yhat'] / forecast['yhat'].iloc[0])
    forecast['yhat_lower'] = initial_price * (forecast['yhat_lower'] / forecast['yhat_lower'].iloc[0])
    forecast['yhat_upper'] = initial_price * (forecast['yhat_upper'] / forecast['yhat_upper'].iloc[0])

    # Set the historical data range (last 3 months)
    three_months_ago = datetime.now() - timedelta(days=90)
    historical_data = stock_data[stock_data.index >= three_months_ago]

    # Plot the forecast with uncertainty intervals
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(historical_data.index, historical_data['Open'], label='Historical Open Prices')
    ax.plot(forecast['ds'], forecast['yhat'], label='Predicted Open Prices')
    ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], alpha=0.2, label='Prediction Interval')
    ax.axvline(datetime.now(), linestyle='--', color='gray', label='Current Date')
    ax.axhline(latest_price, linestyle='--', color='red', label='Latest Price')
    ax.set_xlabel("Date")
    ax.set_ylabel("Open Price")
    ax.set_title(f"{symbol} Stock Open Price Prediction (Next 365 Days)")

    # Set x-axis limits
    ax.set_xlim(three_months_ago, pd.to_datetime(prediction_end_date))

    # Set x-axis format to display months
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.xticks(rotation=45)

    # Add grid lines
    ax.grid(True, linestyle='--', alpha=0.7)

    ax.legend()
    plt.tight_layout()
    plt.show()

    # Plot the forecast components
    fig2 = best_model.plot_components(forecast)
    plt.show()

def main():
    symbol = None
    while True:
        if symbol:
            print(f"Current stock symbol: {symbol}")
        else:
            symbol = input("Enter stock symbol: ").upper()  # Capitalize user's input

        print("\nOptions:")
        print("1. Display basic properties")
        print("2. Display stock information")
        print("3. Retrieve option chain")
        print("4. Change stock symbol")
        print("5. Analyze trends and predict future prices")
        print("6. Exit")

        choice = input("Enter your choice (1-6): ")

        if choice == '1':
            # Retrieve stock data
            data = get_stock_data(symbol)
            # Display basic properties if data is not None
            if data is not None:
                display_basic_properties(data)
            else:
                print("Unable to retrieve data.")
        elif choice == '2':
            # Retrieve stock information
            info = get_stock_info(symbol)
            # Display stock information if info is not None
            if info is not None:
                display_stock_info(info)
            else:
                print("Unable to retrieve data.")
        elif choice == '3':
            # Prompt user for expiration date
            expiration_date = input("Enter expiration date (YYYY-MM-DD): ")
            # Retrieve option chain
            option_chain = get_option_chain(symbol, expiration_date)
            # Display option chain data if option_chain is not None
            if option_chain is not None:
                print("Option Chain Data:")
                print("Calls:")
                print(option_chain.calls)
                print("\nPuts:")
                print(option_chain.puts)
            else:
                print("Unable to retrieve option chain.")
        elif choice == '4':
            symbol = None  # Reset the stock symbol so user can enter a new one
        elif choice == '5':
            # Call the analyze_and_predict function with the current symbol
            analyze_and_predict(symbol)
        elif choice == '6':
            print("Exiting the program. Thank you!")
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 6.")

if __name__ == "__main__":
    main()
    