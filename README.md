# Stock Price Prediction Using Machine Learning

Stock price prediction involves forecasting the future price of a company's stock based on historical data and other influencing factors. Machine learning techniques can be leveraged to make predictions by identifying patterns and relationships in the data.

## Table of Contents
- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Data Collection](#data-collection)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Feature Engineering](#feature-engineering)
- [Model Selection](#model-selection)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Future Improvements](#future-improvements)
- [Conclusion](#conclusion)

## Introduction
Stock price prediction is a challenging task due to market volatility and the influence of numerous external factors. However, machine learning models can provide insights by analyzing historical data and making informed predictions.

## Prerequisites
- Python programming knowledge
- Familiarity with libraries such as Pandas, NumPy, Matplotlib, and Scikit-learn
- Understanding of supervised learning techniques

## Data Collection
- Use financial APIs (e.g., Alpha Vantage, Yahoo Finance, or Quandl) to collect historical stock price data.
- Collect features like Open, High, Low, Close, Volume, and date.

```python
import yfinance as yf

data = yf.download('AAPL', start='2015-01-01', end='2025-01-01')
data.reset_index(inplace=True)
print(data.head())
```

## Data Preprocessing
1. Handle missing values by using techniques like interpolation or forward-fill.
2. Normalize numerical features to scale the data.
3. Convert dates to a numerical format if needed.

```python
# Handle missing values
data.fillna(method='ffill', inplace=True)

# Normalize features
data[['Open', 'High', 'Low', 'Close', 'Volume']] = (
    data[['Open', 'High', 'Low', 'Close', 'Volume']] - data[['Open', 'High', 'Low', 'Close', 'Volume']].mean()
) / data[['Open', 'High', 'Low', 'Close', 'Volume']].std()
```

## Exploratory Data Analysis (EDA)
- Visualize stock trends using line charts.
- Analyze correlations between features.
- Identify seasonality and trends.

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(data['Date'], data['Close'], label='Close Price')
plt.title('Stock Price Over Time')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()
```

## Feature Engineering
- Create new features such as moving averages, relative strength index (RSI), or Bollinger Bands.
- Lag features to capture the sequential nature of stock data.

```python
# Create a moving average feature
data['MA50'] = data['Close'].rolling(window=50).mean()
data['MA200'] = data['Close'].rolling(window=200).mean()
```

## Model Selection
- Common models used for stock price prediction:
  - Linear Regression
  - Random Forest Regressor
  - Gradient Boosting Machines
  - LSTMs (for sequential data)

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

X = data[['Open', 'High', 'Low', 'Volume', 'MA50', 'MA200']].dropna()
y = data['Close'][X.index]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor()
```

## Model Training
Train the model using the training dataset and evaluate it on the testing dataset.

```python
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

## Model Evaluation
Evaluate the model's performance using metrics such as:
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- R-squared

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"R2 Score: {r2}")
```

## Future Improvements
- Include external factors like news sentiment, financial reports, or economic indicators.
- Experiment with deep learning models like LSTMs or Transformers.
- Optimize hyperparameters using techniques like Grid Search or Random Search.

## Conclusion
This project demonstrates how machine learning models can be utilized to predict stock prices. While predictions are never 100% accurate, these techniques can provide valuable insights into market trends and assist in decision-making.

## contributer
[srujan rana](https://github.com/Srujanrana07)
