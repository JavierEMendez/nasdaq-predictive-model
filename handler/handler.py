#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import datetime as dt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import boto3
from os import environ as env

from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, classification_report


def fetch_data(path):
    nasdaq = pd.read_csv(path)
    nasdaq_2 = nasdaq[nasdaq['Close']!= 'Close']
    nasdaq_2['Close2'] = nasdaq_2['Close'].astype(float)
    nasdaq_2.drop(columns=['Open', 'High', 'Low', 'Volume', 'Close'], inplace=True)
    nasdaq_2.set_index(['Symbol', 'Date'], inplace =True)
    nasdaq_2 = nasdaq_2.groupby('Symbol').pct_change()
    nasdaq_2 = nasdaq_2.dropna()
    nasdaq_2 = nasdaq_2.rename(columns={'Close2': 'Returns'})
    nasdaq_2 = nasdaq_2.sort_values('Date')
    return nasdaq_2



def trading_signals(nasdaq_2, short_window=1, long_window=10, short_vol_window=1, long_vol_window=10, bollinger_window=20):

    # Construct a `Fast` and `Slow` Exponential Moving Average from short and long windows, respectively
    nasdaq_2['fast_close'] = nasdaq_2['Returns'].ewm(halflife=short_window).mean()
    nasdaq_2['slow_close'] = nasdaq_2['Returns'].ewm(halflife=long_window).mean()

    # Construct a crossover trading signal
    nasdaq_2['crossover_long'] = np.where(nasdaq_2['fast_close'] > nasdaq_2['slow_close'], 1.0, 0.0)
    nasdaq_2['crossover_short'] = np.where(nasdaq_2['fast_close'] < nasdaq_2['slow_close'], -1.0, 0.0)
    nasdaq_2['crossover_signal'] = nasdaq_2['crossover_long'] + nasdaq_2['crossover_short']

    # Construct a `Fast` and `Slow` Exponential Moving Average from short and long windows, respectively
    nasdaq_2['fast_vol'] = nasdaq_2['Returns'].ewm(halflife=short_vol_window).std()
    nasdaq_2['slow_vol'] = nasdaq_2['Returns'].ewm(halflife=long_vol_window).std()

    # Construct a crossover trading signal
    nasdaq_2['vol_trend_long'] = np.where(nasdaq_2['fast_vol'] < nasdaq_2['slow_vol'], 1.0, 0.0)
    nasdaq_2['vol_trend_short'] = np.where(nasdaq_2['fast_vol'] > nasdaq_2['slow_vol'], -1.0, 0.0) 
    nasdaq_2['vol_trend_signal'] = nasdaq_2['vol_trend_long'] + nasdaq_2['vol_trend_short']
    
    # Calculate rolling mean and standard deviation
    nasdaq_2['bollinger_mid_band'] = nasdaq_2['Returns'].rolling(window=bollinger_window).mean()
    nasdaq_2['bollinger_std'] = nasdaq_2['Returns'].rolling(window=20).std()

    # Calculate upper and lowers bands of bollinger band
    nasdaq_2['bollinger_upper_band']  = nasdaq_2['bollinger_mid_band'] + (nasdaq_2['bollinger_std'] * 1)
    nasdaq_2['bollinger_lower_band']  = nasdaq_2['bollinger_mid_band'] - (nasdaq_2['bollinger_std'] * 1)

    # Calculate bollinger band trading signal
    nasdaq_2['bollinger_long'] = np.where(nasdaq_2['Returns'] < nasdaq_2['bollinger_lower_band'], 1.0, 0.0)
    nasdaq_2['bollinger_short'] = np.where(nasdaq_2['Returns'] > nasdaq_2['bollinger_upper_band'], -1.0, 0.0)
    nasdaq_2['bollinger_signal'] = nasdaq_2['bollinger_long'] + nasdaq_2['bollinger_short']
    
    # Set x variable list of features
    x_var_list = ['crossover_signal', 'vol_trend_signal', 'bollinger_signal']

    # Shift DataFrame values by 1
    nasdaq_2[x_var_list] = nasdaq_2[x_var_list].shift(1)

    # Drop NAs and replace positive/negative infinity values
    nasdaq_2.dropna(subset=x_var_list, inplace=True)
    nasdaq_2.dropna(subset=['Returns'], inplace=True)
    nasdaq_2 = nasdaq_2.replace([np.inf, -np.inf], np.nan)
    
    # Construct the dependent variable where if daily return is greater than 0, then 1, else, 0.
    # Changing the target variable away from 0 introduces the problem of an imbalanced dataset
    nasdaq_2.dropna(inplace=True)
    
    return nasdaq_2

def target_list(nasdaq_2):

    nasdaq_3 = nasdaq_2.copy()
   
    # Creating a target list
    target_list = []
    stoprw = len(nasdaq_3)
    for rw in range(stoprw):
        if rw != stoprw:
            rwn = rw  
            if nasdaq_3['Returns'][rwn]> 0.02:
                target_list.append(1)
            else:
                target_list.append(0)
        else: 
            print("hit!")
            target_list.append(0)
    nasdaq_3['Target'] = target_list
    return nasdaq_3

def standardize(nasdaq_3):
    
    #Create DF with Returns
    nasdaq_4 = nasdaq_3.drop(['Returns','Positive Return'], axis=1, inplace=True)
    nv = ['fast_close', 'slow_close', 'fast_vol','slow_vol', 'bollinger_mid_band', 'bollinger_std', 'bollinger_upper_band']
    nas = nasdaq_4.copy()
    stdrd = nasdaq_4[nv]
    
    #standardizing using min max scaler
    minmax = preprocessing.MinMaxScaler()
    x_unstdrd = stdrd.values
    x_stdrd = minmax.fit_transform(x_unstdrd)
    nas[nv] = x_stdrd
    return nas

def model(nas, training_start = '2019-Jun-01',training_end = '2020-Apr-29',testing_start =  '2020-Apr-30',testing_end = '2020-Jul-24'):
    
    X = nas.drop(['Target'], axis=1, inplace=True)
    y = nas['Target']
    # Construct the X_train and y_train datasets
    X_train = X.iloc[0:660102]
    y_train = y.iloc[0:660102]
    # Constructing X_test and y_test
    X_test = X.iloc[660103:]
    y_test = y.iloc[660103:]

    model = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=0)
    model.fit(X_train, y_train)

    # Make a prediction of "y" values from the X_test dataset
    predictions = model.predict(X_test)

    # Assemble actual y data (Y_test) with predicted y data (from just above) into two columns in a dataframe:
    Results = y_test.to_frame()
    Results["Predicted Value"] = predictions
    filtered_df = Results[Results['Predicted Value']>0]
    return filtered_df

def create_message(df):
    big_str = """These are today's buys: \n"""
    for index, row in df.iterrows():
        big_str += str(row['Symbol']) + ": " + str(row['Predicted Value']) + "\n"
    return big_str

def publish_message_sns(message):
    """
    :param message: str: message to be sent to SNS
    :return: None
    """
    sns_arn = env.get('SNS_ARN').strip()
    sns_client = boto3.client('sns')
    try:
        response = sns_client.publish(
            TopicArn=sns_arn,
            Message=message
        )
    except Exception as e:
        print(f"ERROR PUBLISHING MESSAGE TO SNS: {e}")

def handler(event, context):
    # This is where all the other functions are called. 
    df = fetch_data('final-model-data.csv')
    trading_signals_df = trading_signals(df)
    target_list_df = target_list(trading_signals_df)
    standardize_df = standardize(target_list_df)
    model_df = model(standardize_df)
    message = create_message(model_df)
    publish_message_sns(message)
    return message
    