Stock Price NeuroEvolution prediction
=====================================

Goal: Train a Neural Network using Genetic Algorithms to "Day Trade" highly liquid stocks or indexes.
Indexes: S&P500, Dow Jones, NASDAQ Composite
Stocks: AAPL, IBM, MSFT, GOOGL

v1 - Accurately predict minute price data. Train to maximize accuracy.
v2 - Do not attempt to predict data. Rather, allow the network to figure out when to buy and sell. Train to maximize profit.

WINDOW = 10 //Try 10-15
DATA = 5 //CLOSE, HIGH, LOW, OPEN, VOLUME

Inputs [413 = 6 + 4*(2*WINDOW*DATA + 1) + 3]
======
Day[5] - Binary values representing the current day of the week
Minute - Minute of the day to predict / Total minutes in day (390)
Index data [3] (eg. S&P, DOW, NASDAQ)
  index_day[WINDOW * DATA] - Index data for the past WINDOW days
  index_min[WINDOW * DATA] - Index data for the past WINDOW minutes
  index_open - Index open for current day
Stock data
  stock_day[WINDOW * DATA] - Stock data for the past WINDOW days
  stock_min[WINDOW * DATA] - Stock data for the past WINDOW minutes
  stock_open - Stock open for current day
budget - borrowing/investing limit
spent - percent of budget currently invested
shares - number of shares owned (percentage of amount possible)

Hidden [250]

Output [2]
======
Buy - number of shares to buy (percentage of amount possible)
Sell - percent of shares to sell (round up)
