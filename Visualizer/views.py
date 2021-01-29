from django.shortcuts import render
from django.http import HttpResponse
from django.contrib import messages


# Finance API, and visualization libraries
import numpy as np
import pandas as pd
import math
import yfinance as yf


# Machine Learning libraries
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import load_model

from sklearn.preprocessing import MinMaxScaler, normalize
from sklearn.preprocessing import MinMaxScaler
from .saved_model.my_model import *
from .marketwatch_webscrape import scrape

PATH = "Visualizer//saved_model//my_model"
stock_model = load_model(PATH) # Load model


## HELPER FUNCTIONS ##

def clean_data(tickers, period, interval):

    """
    Returns requested stock market dataframe
    from yfinance API; This function cleans and parses market data
    to ensure that it is ready for JavaScript backend
    """

    data = yf.download(tickers=tickers, period=period, interval = interval).apply(lambda i: round(i, 2), axis=1)
    dates = list(data.index.strftime('%Y-%m-%d'))
    adjusted_close = list(data['Adj Close'])
    volume = list(data['Volume'])
    len_data = len(adjusted_close)

    ## RSI Preparation ###
    delta = data['Adj Close'].diff(1) # Get the difference in price from the previous data
    delta = delta.dropna()

    # Get positive gains (up) and negative gains (down)
    up = delta.copy()
    down = delta.copy()

    up[up < 0] = 0  # only pos values
    down[down > 0] = 0  # only neg values

    # Get the time period (14 days)
    period = 14
    avg_gain = up.rolling(window=period).mean()    # Calculate Average Gain and Average Loss
    avg_loss = abs(down.rolling(window=period).mean())

    # Calculate the Relative Strength (RS)
    RS = avg_gain/avg_loss

    # Calculate RSI

    RSI = 100.0 - (100.0/ (1 + RS))
    RSI = RSI.dropna()
    rsi_adjusted_close = list(data['Adj Close'][RSI.index.strftime('%Y-%m-%d').tolist()[0]:].values)

    ## RNN Preparation ###
    stock_data_to_df = data['Adj Close'].to_frame()
    stock_data = stock_data_to_df.values

    test_idx = int(round(len(stock_data) * 0.7))
    test = stock_data[test_idx:,:]
    last_30percent_dates = dates[test_idx:]



    scaler = MinMaxScaler()
    scaled_adj_close_prices = scaler.fit_transform(stock_data)
    seq_len = 50

    testset = stock_data_to_df[len(stock_data_to_df) - len(test)- seq_len:].values.reshape(-1, 1)
    testset = scaler.transform(testset)

    X_test, y_test = [], []

    for i in range(seq_len, len(test)+seq_len):
        X_test.append(testset[i-seq_len:i, 0])
        y_test.append(testset[i, 0])

    X_test, y_test = np.array(X_test), np.array(y_test)
    X_test = X_test.reshape((X_test.shape[0],X_test.shape[1],1))


    predictions = stock_model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    true = scaler.inverse_transform(y_test.reshape(len(test), 1))
    list_predictions = predictions.flatten().tolist()
    list_true = true.flatten().tolist()

    temp_dict = {'ticker': tickers,
                 'adjusted_close': adjusted_close,
                 'volume': volume,
                 'predictions': list_predictions,
                 'true_values': list_true,
                 'RSI_data': RSI.values.tolist(),
                 'RSI_dates': RSI.index.strftime('%Y-%m-%d').tolist(),
                 'RSI_adj': rsi_adjusted_close,
                 'dates': dates,
                 'model_dates': last_30percent_dates,
                 'length': len_data
                }

    return temp_dict




def get_data(tickers, period, interval):
    """
    Returns requested stock market dataframe
    from yfiance API
    """
    tickers_list = tickers.split()

    if len(tickers_list) == 1:
        dataset1 = clean_data(tickers_list[0], period = period, interval = interval)
        dataset1['Num_Tickers'] = 1

        dataset2 = {} # dummy dictionary

        for i in dataset1:  # need a more efficient way to update 'dummy keys', javascript seems to execute all conditionals at once
            dataset2[i + str(2)] = [0]

        dataset1.update(dataset2)

        return dataset1

    if len(tickers.split()) == 2:
        first_stock = tickers_list[0]
        second_stock = tickers_list[1]
        dataset1 = clean_data(first_stock, period = period, interval = interval)
        dataset2 = clean_data(second_stock, period = period, interval = interval)

        for i in dataset2:  # replace dict keys to differentiate data
            dataset2[i + str(2)] = dataset2.pop(i)

        dataset1.update(dataset2)
        dataset1['Num_Tickers'] = 2  # add a 'Num_Tickers' key so that JavaScript can identity when to plot two or more graphs on same canvas

        return dataset1


# Create your views here.

## RENDERING FUNCTIONS ##

article_scrape = scrape()

def home(request):
    context = {
        'first_article': article_scrape[0],
        'second_article': article_scrape[1],
        'third_article': article_scrape[2]
    }
    # pass in a variable for html template to use
    return render(request,'Visualizer/home.html', context)


def about(request):
    return render(request, 'Visualizer/about.html', {'title':'About'})


def graph(request):
    # weird ASCII characters show up when parsing... fixed
    return render(request,'Visualizer/graph.html', context=get_data('AAPL', '10y', '1d'))

def search(request):

    search_query = request.GET.get('ticker_search')


    if request.method == 'POST':
        user_period = request.POST.get('period')
        user_interval = request.POST.get('interval')
        add_stock_query = request.POST.get('add_stock')

        if add_stock_query == '':
            return render(request,'Visualizer/about.html', {'ticker': add_stock_query,'error':'Enter at least one ticker.'})

        else:
            try:
                get_data(add_stock_query, user_period, user_interval)

            except:
                return render(request,'Visualizer/about.html', {'ticker': add_stock_query,'error':'Please enter a different ticker.'})

            else:
                return render(request, 'Visualizer/ticker_search.html', context=get_data(add_stock_query, user_period, user_interval))


    if search_query:  # GET
        try:
            get_data(search_query, '10y', '1d')

        except:
            return render(request,'Visualizer/about.html', {'ticker': search_query,'error':'Please enter a different ticker.'})

        else:
            return render(request, 'Visualizer/ticker_search.html', context=get_data(search_query, '10y', '1d'))
