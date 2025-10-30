import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import random


nifty_50_tickers = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "HINDUNILVR.NS",
    "ICICIBANK.NS", "SBIN.NS", "BAJFINANCE.NS", "BHARTIARTL.NS", "ASIANPAINT.NS",
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


def compute_metrics(tickers, end_date=None, rf=0.03):
    """
    Return a DataFrame with metrics as rows and tickers as columns.
    Metrics include short returns, 1/3/5Y CAGR, annualized volatility, annualized return (1Y),
    and Sharpe ratio (annualized).
    
    Parameters:
    -----------
    tickers : list
        List of ticker strings (yfinance format)
    end_date : str or pd.Timestamp, optional
        End date for analysis
    rf : float, default 0.03
        Risk-free rate (annual, decimal)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with metrics as rows and tickers as columns
    """
    end = pd.to_datetime(end_date) if end_date is not None else pd.Timestamp.today()
    start = end - pd.DateOffset(years=6) - pd.Timedelta(days=10)
    # download adjusted close prices for all tickers
    # Download data for each ticker separately to handle failures
    all_data = pd.DataFrame()
    valid_tickers = []
    
    for ticker in tickers:
        try:
            data = yf.download(ticker, 
                             start=start.strftime("%Y-%m-%d"),
                             end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                             progress=False)
            
            # Try Adj Close first, fall back to Close if needed
            if 'Adj Close' in data.columns:
                price_series = data['Adj Close']
            elif 'Close' in data.columns:
                price_series = data['Close']
            else:
                raise ValueError("Neither Adj Close nor Close column found")
                
            if not price_series.empty:
                all_data[ticker] = price_series
                valid_tickers.append(ticker)
                print(f"Successfully downloaded data for {ticker}")
        except Exception as e:
            print(f"Warning: Could not download data for {ticker}: {str(e)}")
    
    if all_data.empty:
        raise RuntimeError("No price data could be downloaded for any tickers. Try different stocks.")
    
    data = all_data.dropna(how="all")
    tickers = valid_tickers  # Update tickers list to only include valid ones

    def price_at(series, target_dt):
        subset = series[series.index <= target_dt]
        return subset.iloc[-1] if not subset.empty else np.nan

    present_date = data.dropna(how="all").index[-1]
    metrics = {}
    short_periods = {"1-Day": 1, "5-Day": 5, "1-Month": 30, "3-Month": 90, "6-Month": 182}

    for t in data.columns:
        s = data[t].dropna()
        if s.empty:
            col = {k: np.nan for k in list(short_periods.keys()) + ["1Y CAGR", "3Y CAGR", "5Y CAGR",
                                                                   "Ann Vol (%)", "1Y Return (%)", "Sharpe (ann)"]}
            metrics[t] = col
            continue

        present_price = s.iloc[-1]
        row = {}
        # short returns
        for name, days in short_periods.items():
            target = present_date - pd.Timedelta(days=days)
            past = price_at(s, target)
            row[name] = ((present_price / past - 1) * 100.0) if pd.notna(past) and past != 0 else np.nan

        # CAGR calculations
        def cagr_years(y):
            target = present_date - pd.DateOffset(years=y)
            subset = s[s.index <= target]
            if subset.empty:
                return np.nan
            start_price = subset.iloc[-1]
            actual_years = (present_date - subset.index[-1]).days / 365.25
            if actual_years <= 0 or start_price == 0:
                return np.nan
            return ((present_price / start_price) ** (1.0 / actual_years) - 1) * 100.0

        row["1Y CAGR"] = cagr_years(1)
        row["3Y CAGR"] = cagr_years(3)
        row["5Y CAGR"] = cagr_years(5)

        # Volatility and returns metrics
        daily_ret = s.pct_change().dropna()
        ann_vol = daily_ret.std() * np.sqrt(252) * 100.0
        r1 = row["1Y CAGR"]
        
        if pd.isna(r1):
            target = present_date - pd.DateOffset(years=1)
            subset = s[s.index <= present_date]
            subset2 = subset[subset.index >= target]
            if not subset2.empty:
                yrs = (present_date - subset2.index[0]).days / 365.25
                r1 = ((present_price / subset2.iloc[0]) ** (1.0 / yrs) - 1) * 100.0 if yrs > 0 else np.nan

        row["Ann Vol (%)"] = ann_vol
        row["1Y Return (%)"] = r1

        # Sharpe ratio calculation
        ann_ret_dec = (r1 / 100.0) if pd.notna(r1) else np.nan
        ann_vol_dec = (ann_vol / 100.0) if pd.notna(ann_vol) else np.nan
        row["Sharpe (ann)"] = ((ann_ret_dec - rf) / ann_vol_dec) if (pd.notna(ann_ret_dec) and pd.notna(ann_vol_dec) and ann_vol_dec != 0) else np.nan

        metrics[t] = row

    df = pd.DataFrame(metrics)
    row_order = list(short_periods.keys()) + ["1Y CAGR", "3Y CAGR", "5Y CAGR", "1Y Return (%)", "Ann Vol (%)", "Sharpe (ann)"]
    return df.reindex(row_order)


def plot_comparisons(tickers, end_date=None):
    """
    Create comprehensive comparison plots for multiple stocks.
    
    Parameters:
    -----------
    tickers : list
        List of ticker strings (yfinance format)
    end_date : str or pd.Timestamp, optional
        End date for analysis
    
    Returns:
    --------
    None
        Displays multiple plots:
        - Cumulative returns (normalized)
        - CAGR comparison (1Y/3Y/5Y)
        - Risk-Return scatter plot
        - Correlation heatmap
    """
    # Ensure matplotlib is in interactive mode
    plt.ion()
    end = pd.to_datetime(end_date) if end_date is not None else pd.Timestamp.today()
    start = end - pd.DateOffset(years=6) - pd.Timedelta(days=10)
    
    # Download data for each ticker separately
    prices = pd.DataFrame()
    for ticker in tickers:
        try:
            data = yf.download(ticker, 
                             start=start.strftime("%Y-%m-%d"),
                             end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                             progress=False)
            if 'Adj Close' in data.columns:
                prices[ticker] = data['Adj Close']
            else:
                prices[ticker] = data['Close']
        except Exception as e:
            print(f"Warning: Could not download data for {ticker}: {str(e)}")
            
    if prices.empty:
        raise RuntimeError("No price data could be downloaded for any tickers")
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0])
    
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0])
    prices = prices.dropna(how="all")
    if prices.empty:
        raise RuntimeError("No price data for plotting")

    # Create figure with subplots
    fig = plt.figure(figsize=(15, 12))
    
    # 1. Cumulative returns
    ax1 = plt.subplot(221)
    norm = prices.divide(prices.iloc[0])
    for col in norm.columns:
        ax1.plot(norm.index, norm[col], label=col)
    ax1.set_title("Cumulative Returns (Normalized)")
    ax1.legend(ncol=2, fontsize='small', bbox_to_anchor=(1.05, 1))
    ax1.grid(alpha=0.3)
    
    # 2. CAGR Comparison
    ax2 = plt.subplot(222)
    metrics = compute_metrics(tickers, end_date=end)
    cagr_cols = ["1Y CAGR", "3Y CAGR", "5Y CAGR"]
    cagr_data = metrics.loc[cagr_cols] if all(col in metrics.index for col in cagr_cols) else pd.DataFrame()
    if not cagr_data.empty:
        cagr_data.T.plot(kind='bar', ax=ax2)
        ax2.set_title("CAGR Comparison")
        ax2.set_ylabel("CAGR (%)")
        ax2.legend(title="Period", fontsize='small', bbox_to_anchor=(1.05, 1))
        plt.setp(ax2.get_xticklabels(), rotation=45)
    else:
        ax2.text(0.5, 0.5, "Insufficient data for CAGR calculation",
                ha='center', va='center')
        ax2.set_title("CAGR Comparison (N/A)")
    
    # 3. Risk-Return Scatter
    ax3 = plt.subplot(223)
    try:
        vol = metrics.loc["Ann Vol (%)"]
        ret = metrics.loc["1Y Return (%)"]
        ax3.scatter(vol, ret)
        for i, t in enumerate(vol.index):
            ax3.annotate(t, (vol.iloc[i], ret.iloc[i]))
        ax3.set_xlabel("Annualized Volatility (%)")
        ax3.set_ylabel("1Y Return (%)")
        ax3.set_title("Risk-Return Profile")
        ax3.grid(alpha=0.3)
    except:
        ax3.text(0.5, 0.5, "Insufficient data for risk-return analysis",
                ha='center', va='center')
        ax3.set_title("Risk-Return Profile (N/A)")
    
    # 4. Correlation Heatmap
    ax4 = plt.subplot(224)
    daily_returns = prices.pct_change().dropna()
    corr = daily_returns.corr()
    im = ax4.imshow(corr, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax4)
    ax4.set_xticks(range(len(corr.columns)))
    ax4.set_yticks(range(len(corr.index)))
    ax4.set_xticklabels(corr.columns, rotation=45, ha='right')
    ax4.set_yticklabels(corr.index)
    ax4.set_title("Correlation Matrix")
    
    # Adjust layout and display
    plt.tight_layout()
    plt.show()
    
    return metrics   