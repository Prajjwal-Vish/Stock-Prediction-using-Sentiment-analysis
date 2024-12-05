# predict_ohl.py
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Function to predict Open price
def predict_open(prices):
    open_prices = prices['Open'][-7:].values.reshape(-1, 1)
    days = np.array(range(7)).reshape(-1, 1)
    
    model = LinearRegression()
    model.fit(days, open_prices)
    next_day = np.array([[7]])  # Predict the 8th day's Open
    predicted_open = model.predict(next_day)
    
    return predicted_open[0][0]

# Function to predict Low price
def predict_low(prices):
    low_prices = prices['Low'][-7:].values.reshape(-1, 1)
    days = np.array(range(7)).reshape(-1, 1)
    
    model = LinearRegression()
    model.fit(days, low_prices)
    next_day = np.array([[7]])  # Predict the 8th day's Low
    predicted_low = model.predict(next_day)
    
    return predicted_low[0][0]

# Function to predict High price
def predict_high(prices):
    high_prices = prices['High'][-7:].values.reshape(-1, 1)
    days = np.array(range(7)).reshape(-1, 1)
    
    model = LinearRegression()
    model.fit(days, high_prices)
    next_day = np.array([[7]])  # Predict the 8th day's High
    predicted_high = model.predict(next_day)
    
    return predicted_high[0][0]

# Function to predict Volume
def predict_volume(prices):
    volume = prices['Volume'][-7:].values.reshape(-1, 1)
    days = np.array(range(7)).reshape(-1, 1)
    
    model = LinearRegression()
    model.fit(days, volume)
    next_day = np.array([[7]])  # Predict the 8th day's Volume
    predicted_volume = model.predict(next_day)
    
    return predicted_volume[0][0]

# Function to predict Sentiment Score
def predict_sentiment(prices):
    sentiment_scores = prices['sentiment_score'][-7:].values.reshape(-1, 1)
    days = np.array(range(7)).reshape(-1, 1)
    
    model = LinearRegression()
    model.fit(days, sentiment_scores)
    next_day = np.array([[7]])  # Predict the 8th day's sentiment_score
    predicted_sentiment = model.predict(next_day)
    
    return predicted_sentiment[0][0]

# Function to predict Tweet Count
def predict_tweet_count(prices):
    tweet_counts = prices['tweet_count'][-7:].values.reshape(-1, 1)
    days = np.array(range(7)).reshape(-1, 1)
    
    model = LinearRegression()
    model.fit(days, tweet_counts)
    next_day = np.array([[7]])  # Predict the 8th day's tweet count
    predicted_tweet_count = model.predict(next_day)
    
    return predicted_tweet_count[0][0]

def calculate_rsi(df, period=14):

    # Compute the differences between consecutive closing prices
    delta = df['Close'].diff()

    # Gain and loss for each day
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)

    # Calculate the average gain and loss over the last 'period' days
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    # Calculate the RSI
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # Return the RSI value for the current (last) row
    return rsi.iloc[-1]


## Function to predict adx 
def calculate_adx(df, period=14):
    # Calculate the True Range (TR)
    high_low = df['High'] - df['Low']
    high_close = abs(df['High'] - df['Close'].shift())
    low_close = abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    # Calculate the Directional Movement (+DM and -DM)
    plus_dm = df['High'].diff()
    minus_dm = -df['Low'].diff()

    plus_dm = plus_dm.where(plus_dm > 0, 0)
    minus_dm = minus_dm.where(minus_dm > 0, 0)

    # Smooth the TR and DM values over the period
    tr_smooth = tr.rolling(window=period).sum()
    plus_dm_smooth = plus_dm.rolling(window=period).sum()
    minus_dm_smooth = minus_dm.rolling(window=period).sum()

    # Calculate the +DI and -DI
    plus_di = (plus_dm_smooth / tr_smooth) * 100
    minus_di = (minus_dm_smooth / tr_smooth) * 100

    # Calculate the ADX
    adx = abs(plus_di - minus_di)
    adx = adx.rolling(window=period).mean()

    # Return the ADX value for the current (last) row
    return adx.iloc[-1]  # This returns the ADX value for the most recent day


def calculate_atr(df, period=14):
    # Calculate the True Range (TR)
    high_low = df['High'] - df['Low']
    high_close = abs(df['High'] - df['Close'].shift())
    low_close = abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    # Return the ATR value for the current (last) row, using the rolling mean over 'n' days
    return tr.rolling(window=period).mean().iloc[-1]  # This returns the ATR value for the most recent day



def calculate_sma(df, period):
    sma = df['Close'].rolling(window=period).mean()

    # Return the SMA value for the current (last) row
    return sma.iloc[-1]  # This returns the SMA value for the most recent day
