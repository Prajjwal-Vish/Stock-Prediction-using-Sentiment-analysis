# get_predictions.py
import pandas as pd
import numpy as np
import talib as ta
from datetime import timedelta
from predict_ohl import (predict_open, predict_low, predict_high, predict_volume, predict_sentiment, predict_tweet_count, calculate_rsi, calculate_adx, calculate_atr, calculate_sma)



def get_newrow(df):
    open = predict_open(df)
    low = predict_low(df)
    high = predict_high(df)
    volume = predict_volume(df)
    sentiment_score = predict_sentiment(df)
    tweet_count = predict_tweet_count(df)
    rsi = calculate_rsi(df)
    adx = calculate_adx(df)
    atr = calculate_atr(df)
    sma_3 = calculate_sma(df,3)
    sma_7 = calculate_sma(df,7)
    sma_14 = calculate_sma(df,14)
    last_date = df['Date'].iloc[-1]
    next_date = pd.to_datetime(last_date) + pd.Timedelta(days=1)  
    
    # Create a new row with all the values
    new_row = {
        'Date': next_date,
        'Open': open,
        'Low': low,
        'High': high,
        'Volume': volume,
        'rsi': rsi,
        'adx': adx,
        'atr': atr, 
        'sma_3': sma_3,
        'sma_7': sma_7,
        'sma_14': sma_14,
        'sentiment_score': sentiment_score,
        'tweet_count': tweet_count
    }
    
    
    new_row_df = pd.DataFrame([new_row])
