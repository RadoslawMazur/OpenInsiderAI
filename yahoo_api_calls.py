import pandas as pd
from tqdm import tqdm
import os
import yfinance as yf

def download_price_history(ticker, start_date="2000-01-01", output_dir="data"):
    """
    Downloads historical price data for the given list of tickers since the specified start date.

    Parameters:
        tickers (list): List of ticker symbols.
        start_date (str): The start date for historical data in YYYY-MM-DD format.
        save_to_csv (bool): Whether to save the data as CSV files. Default is True.
        output_dir (str): Directory to save the CSV files if save_to_csv is True. Default is 'historical_data'.

    Returns:
        dict: A dictionary with ticker symbols as keys and their corresponding DataFrame as values.
    """

    try:
        print(f"Downloading data for {ticker}...")
        # Download data using yfinance
        data = yf.download(ticker, start=start_date)

        if data.empty:
            print(f"No data found for {ticker}. Skipping.")
            return None

        # Save to CSV if required
        output_file = os.path.join(output_dir, f"{ticker}_history.csv")
        data.to_csv(output_file)
        print(f"Saved {ticker} data to {output_file}.")

    except Exception as e:
        print(f"Failed to download data for {ticker}. Error: {e}")
        return None

    return ticker, len(data)

tickers = pd.read_csv("oi_csv.csv", usecols=["tick"])["tick"].str.strip(" ").dropna().unique().tolist()
tickers.extend(["SPY", "^DJI"])
avail_ticks = []

for tick in tqdm(tickers):
    checked_tick = download_price_history(tick)

    if checked_tick:
        avail_ticks.append(checked_tick)


pd.DataFrame(avail_ticks, columns=["tick", "days_num"]).to_csv("tick_count.csv")