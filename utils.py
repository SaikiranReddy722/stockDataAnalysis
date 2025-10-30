import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import random


nifty_50_tickers = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "HINDUNILVR.NS",
    "ICICIBANK.NS", "KOTAKBANK.NS", "SBIN.NS", "BAJFINANCE.NS", "HDFC.NS",
    "ITC.NS", "LT.NS", "AXISBANK.NS", "MARUTI.NS", "ASIANPAINT.NS",
    "WIPRO.NS", "SUNPHARMA.NS", "DIVISLAB.NS", "TITAN.NS", "NESTLEIND.NS",
    "ULTRACEMCO.NS", "M&M.NS", "TECHM.NS", "HCLTECH.NS", "POWERGRID.NS",
    "JSWSTEEL.NS", "ADANIGREEN.NS", "BAJAJ-AUTO.NS", "GRASIM.NS", "COALINDIA.NS",
    "SBILIFE.NS", "DRREDDY.NS", "TATASTEEL.NS", "CIPLA.NS", "HEROMOTOCO.NS",
    "EICHERMOT.NS", "BRITANNIA.NS", "HEROIND.NS", "GAIL.NS", "TATAMOTORS.NS",
    "SHREECEM.NS", "ADANIPORTS.NS", "ONGC.NS", "VEDL.NS", "INDUSINDBK.NS", "HDFCLIFE.NS",
    "BPCL.NS", "DLF.NS", "TATACONSUM.NS", "MUTHOOTFIN.NS", "CROMPTON.NS", "LUPIN.NS"
]


def random_stocks(n=5):
    """Select n random stocks from the Nifty 50 list."""
    return random.sample(nifty_50_tickers, n)   