# StockViz : https://stockviz-123.herokuapp.com/
A Django web application that allows users to visualize and analyze stock data. Users can interact with graphs, compare stocks against each other and use various technical indicators
(**Bollinger Bands** and **RSI Plots**) to determine market trends.


# Contributors:
*Shahriyar Habib* 
*Don Min*


# Libraries
The entire list of required libraries can be viewed in **requirements.txt**

**Tensorflow**:
   *  Created a RNN (LSTM-based) model that is trained with time-series data; allows users to test our model's performance against adjusted closing data

**Django/bs4**:
   *  Framework for building medium-large scale web applications in python; used the beautifulsoup library to extract data from marketwatch.com to obtain live article data for our web app

**yfinance**: 
   * Stock market API that allows users to call various features (adj_close, volume, close, etc.) of stocks; converts the data into a pandas dataframe


Deployed on Heroku.
