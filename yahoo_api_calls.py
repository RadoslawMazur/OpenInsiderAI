import requests
import pandas as pd
import time


tickers = pd.read_csv("oi_csv_2.csv", usecols=["tick"])["tick"].str.strip(" ").dropna().unique().tolist()
tickers.extend(["SPY", "^DJI"])
headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/113.0"}
avail_ticks = []

for tick in tickers:
    yahoo_url = f"https://query1.finance.yahoo.com/v7/finance/download/{tick}?period1=-10000000000&period2=991686441600&interval=1d&events=history&includeAdjustedClose=true"
    try:
        r = requests.get(yahoo_url, headers=headers)
        content = r.content.decode()
        with open(f"data/{tick}.csv", "w") as f:
            f.write(content)
        avail_ticks.append([tick, content.count("\n")])
        time.sleep(10)
    except (requests.RequestException, requests.ConnectionError):
        print(f"Error on tick: {tick}")


pd.DataFrame(avail_ticks, columns=["tick", "days_num"]).to_csv("tick_count.csv")