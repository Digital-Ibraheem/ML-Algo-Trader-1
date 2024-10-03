# ML-Algo-Trader-1

# Algorithmic Trader Using Random Forest with S&P 500

This project is a learning exercise in Machine Learning (ML) using stock market data from the S&P 500. The goal is to explore how ML models, such as a Random Forest Classifier, can be used to predict future price movements. **Please note that the accuracy of this model is only 57%, and it was developed purely for educational purposes.**

**Disclaimer: This algorithm is not financial advice, and I do not recommend anyone to use it for trading or investing decisions.**

## Features

- **Stock Data**: Historical data for the S&P 500 index is fetched using the `yfinance` library.
- **Data Cleaning**: Removes irrelevant columns like dividends and stock splits, focusing on price and volume data.
- **Feature Engineering**:
  - A target variable is created that predicts whether the next day's closing price will be higher or lower than today's.
  - Rolling averages and trends over multiple time horizons (e.g., 2, 5, 60, 250, and 1000 days) are calculated to enhance predictive accuracy.
- **Machine Learning**:
  - A Random Forest Classifier is used for prediction, with options for tuning hyperparameters like the number of estimators and minimum sample split size.
- **Backtesting**: Simulates historical trades to evaluate model performance over time.

## How It Works

1. **Data Fetching**: The script uses `yfinance` to download the S&P 500 index data.
    ```python
    import yfinance as yf
    sp500 = yf.Ticker("^GSPC")
    sp500 = sp500.history(period="max")
    ```

2. **Feature Engineering**:
    - The target variable (`Target`) indicates whether tomorrow's closing price is higher than today's.
    - Rolling averages and trend data are created for multiple horizons (2, 5, 60, 250, 1000 days).
    ```python
    sp500["Tomorrow"] = sp500["Close"].shift(-1)
    sp500["Target"] = (sp500["Tomorrow"] > sp500["Close"]).astype(int)
    ```

3. **Model Training**:
    - A `RandomForestClassifier` is trained using past data. The key predictors are `Close`, `Volume`, `Open`, `High`, and `Low`.
    ```python
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)
    ```

4. **Backtesting**:
    - The `backtest` function evaluates the model's performance over time, using historical data to simulate trading.
    ```python
    def backtest(data, model, predictors, start=2500, step=250):
        all_predictions = []
        for i in range(start, data.shape[0], step):
            train = data.iloc[0:i].copy()
            test = data.iloc[i:(i+step)].copy()
            predictions = predict(train, test, predictors, model)
            all_predictions.append(predictions)
        return pd.concat(all_predictions)
    ```

## Key Functions

- **`predict(train, test, predictors, model)`**: Trains the model on the training data and makes predictions on the test data.
- **`backtest(data, model, predictors)`**: Runs backtesting on the data using the specified model and predictors.

## Model Accuracy

The model achieves an accuracy of **57%**, which is slightly better than random guessing. This result highlights the complexity of stock market prediction and the limitations of simple machine learning models in such a volatile domain.

## Disclaimer

This model was created purely as a learning exercise to understand the world of Machine Learning. **It is not financial advice, and I do not recommend anyone to use this algorithm for making trading or investment decisions.**
