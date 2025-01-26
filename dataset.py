import concurrent.futures
import torch
from typing import List, Optional
import pandas as pd
from time import time_ns
import random
import isoweek
from functools import cache
import os
import pandas_datareader.data as web
import datetime
import concurrent
from tqdm import tqdm


class TradeDataset:

    def _get_us_gdp(self) -> pd.DataFrame:

        df = web.DataReader('GDP', 'fred', self.start_date, self.end_date).reset_index()

        start_day_abs_week = self.start_date.isocalendar()[1] + self._add_weeks_per_year(self.start_date.year)
        end_day_abs_week = self.end_date.isocalendar()[1] + self._add_weeks_per_year(self.end_date.year)
        table_range = range(start_day_abs_week, end_day_abs_week)

        df[["year", "week", "day"]] = pd.to_datetime(df["DATE"]).dt.isocalendar()
        df["abs_week"] = df["week"] + df["year"].apply(self._add_weeks_per_year)
        df = df[df["abs_week"] > 0]
        df = pd.merge(pd.DataFrame(table_range, columns=["abs_week"]), df, how="left")
        df["GDP"] = df["GDP"].ffill()

        for i in [52, 104]:
            df[f"{i}w_GDP_change"] = (df["GDP"] / df["GDP"].shift(i) - 1) * 100

        df = df.bfill()
        
        return df[["abs_week", "52w_GDP_change", "104w_GDP_change"]]
    
    def _get_interest_rate(self) -> pd.DataFrame:

        df = web.DataReader('FEDFUNDS', 'fred', self.start_date, self.end_date).reset_index()

        start_day_abs_week = self.start_date.isocalendar()[1] + self._add_weeks_per_year(self.start_date.year)
        end_day_abs_week = self.end_date.isocalendar()[1] + self._add_weeks_per_year(self.end_date.year)
        table_range = range(start_day_abs_week, end_day_abs_week)
        
        df[["year", "week", "day"]] = pd.to_datetime(df["DATE"]).dt.isocalendar()
        df["abs_week"] = df["week"] + df["year"].apply(self._add_weeks_per_year)
        df = df[df["abs_week"] > 0]
        df = pd.merge(pd.DataFrame(table_range, columns=["abs_week"]), df, how="left").bfill()
        df["FEDFUNDS"] = df["FEDFUNDS"].ffill()

        for i in [26, 52]:
            df[f"{i}w_FEDFUNDS_change"] = (df["FEDFUNDS"] / df["FEDFUNDS"].shift(i) - 1)

        return df[["abs_week", "FEDFUNDS", "26w_FEDFUNDS_change", "52w_FEDFUNDS_change"]]
    
    def _get_market_table(self):
        us_gdp_df = self._get_us_gdp()
        interest_rate = self._get_interest_rate()

        return pd.merge(us_gdp_df, interest_rate, how="inner", on="abs_week")


    @cache # cache is crucial here
    def _add_weeks_per_year(self, year) -> int:
        start_year = 1999
        weeks = 0 if year>=start_year else -999
        for y in range(start_year, year):
            weeks += isoweek.Week.last_week_of_year(y)[1]
        return int(weeks)

    @classmethod 
    def week_to_date(cls, abs_week: int) -> datetime.datetime:

        start_year = 1999
        pass

    @classmethod
    def date_to_week(cls, date: datetime.datetime) -> int:

        year = date.year
        start_year = 1999
        weeks = 0 if year>=start_year else -999
        for y in range(start_year, year):
            weeks += isoweek.Week.last_week_of_year(y)[1]
        return int(weeks) + int(date.isocalendar()[1])

    def _get_insider_trades(self) -> pd.DataFrame:
        df = pd.read_csv("oi_csv.csv", usecols=["Filling", "Trade", "tick", "title", "type", "price", "qty", "owned", "delta_owned", "value"]).drop_duplicates().dropna(subset="tick")

        # TODO here is the place to limit the start and end dates (?????)
        # parse the dates
        df["Trade"] =  pd.to_datetime(df["Trade"])
        df["Filling"] =  pd.to_datetime(df["Filling"]).fillna(df["Trade"])

        df = df[(df["Trade"] > self.start_date) & (df["Trade"] < self.end_date)]

        # filter ticks to those price history is available
        df["tick"] = df["tick"].str.strip(" ")
        df = df[df["tick"].isin(map(lambda x: x[:-len("_history.csv")], os.listdir("data")))]

        # remove ticks with less than 250 transactions (??)
        counts_df = df["tick"].value_counts()
        counts_df = counts_df[counts_df > 250] 
        df = df[df["tick"].isin(counts_df.index)]

        # parse the numbers
        df["price"] = df["price"].str.replace("[$,]", "", regex=True).astype("float32")
        df["qty"] = df["qty"].str.replace(",", "").fillna(0).astype("int32")
        df["owned"] = df["owned"].str.replace(",", "").fillna(0).astype("int32")
        df["delta_owned"] = df["delta_owned"].str.rstrip("%").str.replace("New", "0").str.replace(">999", "1000").fillna(0).astype("int16")
        df["value"] = df["value"].str.replace("[$,]", "", regex=True).fillna(0).astype("int32")


        # get some info from title
        df["is_dir"] = df["title"].str.lower().str.contains("dir").astype(float)
        df["is_ceo"] = df["title"].str.lower().str.contains("ceo").astype(float)
        df["is_major_steakholder"] = df["title"].str.lower().str.contains("10%").astype(float)
        df = df.drop("title", axis=1)

        # days to filling
        df["d_to_filling"] = (df["Trade"] - df["Filling"]).dt.days
        df = df.drop("Filling", axis=1)

        # split trade date
        df[["t_year", "t_week", "t_day"]] = df["Trade"].dt.isocalendar()
        df["abs_week"] = df["t_week"] + df["t_year"].apply(self._add_weeks_per_year)
        df = df.drop(["Trade", "t_year", "t_week", "t_day"], axis=1)

        # one hot encoding of transaction type
        transaction_types = pd.get_dummies(df["type"]) * 1
        df = pd.merge(df, transaction_types, left_index=True, right_index=True)
        df = df.drop("type", axis=1)

        week_df = df.groupby(["tick", "abs_week"]).agg({
            "price": "mean",  # Calculate mean for "value1"
            "qty": "sum",   # Calculate sum for "value2"
            "owned": "sum",    # Calculate max for "value3"
            "delta_owned": "sum",
            "value": "mean",
            "is_dir": "sum",
            "is_ceo": "sum",
            "is_major_steakholder": "sum",
            "d_to_filling": "mean",
            'A - Grant': "sum", 
            'C - Converted deriv': "sum", 
            'D - Sale to issuer': "sum", 
            'F - Tax': "sum",
            'G - Gift': "sum", 
            'M - OptEx': "sum", 
            'P - Purchase': "sum", 
            'S - Sale': "sum", 
            'W - Inherited': "sum",
            'X - OptEx': "sum"

        })

        return week_df.reset_index()
    
    def _get_tick_borders(self):

        df = self.insider_df
        # this is a function to get first and last insider trade for each tick
        df = pd.merge(df.groupby("tick")["abs_week"].max() - self.seq_len - 4, df.groupby("tick")["abs_week"].min(), left_index=True, right_index=True)

        # merge with price data
        price_df = self._get_tick_borders_from_price()
        df = pd.merge(price_df, df, how="inner", left_on="tick", right_index=True)

        df["abs_week_x"] = df[["abs_week_x", "abs_week_x_price"]].min(axis=1).astype(int)
        df["abs_week_y"] = df[["abs_week_y", "abs_week_y_price"]].max(axis=1).astype(int)
        df = df.drop(["abs_week_y_price", "abs_week_x_price"], axis=1)

        df["n_weeks"] = df["abs_week_x"] - df["abs_week_y"]
        df = df[df["n_weeks"] > 10] # removing weeks that do not have ennough weeks


        # calcaulte total weeks
        df["rolling_week"] = df["n_weeks"].cumsum().shift(1).fillna(0).astype(int)
        df = df.reset_index()

        return df
    
    def _get_tick_borders_from_price(self):
        
        tick_borders = [] 
        for tick in self.insider_df["tick"].unique():
            try:
                df = self._get_tick_dataframe(tick)
                tick_borders.append([tick, df["abs_week"].max(), df["abs_week"].min()])
            except KeyError:
                pass

        return pd.DataFrame(tick_borders, columns=["tick", "abs_week_x_price", "abs_week_y_price"])


    @cache 
    def _get_tick_dataframe(self, tick):
        df = pd.read_csv(f"data/{tick}_history.csv", skiprows=[1, 2])

        # get weeks from 1999
        df["Date"] = pd.to_datetime(df["Price"])
        df[["year", "week", "day"]] = df["Date"].dt.isocalendar()
        df["abs_week"] = (df["week"] + df["year"].apply(self._add_weeks_per_year)).astype(int)
        df["mean_price"] = (df["High"] + df["Low"])/2

        # figure out mean week price
        df = df[["abs_week", "year", "week", "mean_price", "Volume"]].groupby(["year", "week"]).mean().reset_index()
        df["abs_week"] = df["abs_week"].astype(int)

        # if there are any missing weeks fill them with previous week
        all_timestamps = pd.DataFrame({"abs_week": range(df["abs_week"].min(), df["abs_week"].max() + 1)})
        df = pd.merge(all_timestamps, df, on="abs_week", how="left")
        df["mean_price"] = df["mean_price"].ffill()
        df["volume"] = df["Volume"].ffill() + 0.01 # to avide 0

        # change in price for X 
        df[f"-1w_change"] = df["mean_price"] / df["mean_price"].shift(1)
        df[f"-1w_change_volume"] = df["Volume"] / df["Volume"].shift(1)

        # calcaulte percent change in price in future weeks
        for i in range(1,5):
            df[f"{i}w_change"] = (df["mean_price"].shift(-i) / df["mean_price"]) - 1

        df = df[df["abs_week"] > 0]

        return df[["abs_week", "-1w_change", "1w_change", "2w_change", "3w_change", "4w_change", "volume", "-1w_change_volume"]]

    def __init__(
            self,
            seq_len: int,
            to_save: bool = False,
            start_date: Optional[datetime.datetime] = None, 
            end_date: Optional[datetime.datetime] = None
        ):

        self.start_date = start_date if start_date else datetime.datetime.strptime("01-01-1999", "%d-%m-%Y")
        self.end_date = end_date if end_date else datetime.datetime.today()

        self.seq_len = seq_len
        self.to_save = to_save
        self.insider_df = self._get_insider_trades()
        self.tick_borders = self._get_tick_borders()
        self.market_table = self._get_market_table()

    def __len__(self) -> int:
        #return (self.insider_df.groupby("tick")["abs_week"].max() - self.insider_df.groupby("tick")["abs_week"].min() - self.seq_len).sum()
        return self.tick_borders["n_weeks"].sum()

    @cache 
    def _get_insider_df_for_tick(self, tick: str) -> pd.DataFrame:
        # This function brings __getitem__ from 20ms to 2ms because merge is much faster
        return self.insider_df[self.insider_df["tick"] == tick]

    def __getitem__(self, index: int) -> List[torch.Tensor]: # 2.5ms

        # select which the tick using rolling week as an index
        week = self.tick_borders.loc[self.tick_borders["rolling_week"] < index+1, "rolling_week"].max()
        tick = self.tick_borders.loc[(self.tick_borders["rolling_week"] == week), "tick"]

        # 
        start_day = self.tick_borders.loc[self.tick_borders["rolling_week"] == week, "abs_week_y"] + index - week
        stop_day = start_day + self.seq_len

        # assetions to confirm 
        assert start_day.size != 0 or stop_day.size != 0 or start_day.size > 2 or stop_day.size > 2, "there should be only one row"
        assert tick.size != 0 or tick.size > 2, "Only one tick should be selected"

        # creating X dataframe
        price_df = self._get_tick_dataframe(tick.iloc[0])

        df = pd.DataFrame(zip([tick.iloc[0]] * self.seq_len, range(start_day.iloc[0], stop_day.iloc[0])), columns=["tick", "abs_week"])
        df = pd.merge(df, self._get_insider_df_for_tick(tick.iloc[0]), on=["tick", "abs_week"], how="left")
        df = pd.merge(df, self.market_table, on=["abs_week"], how="left")
        df = pd.merge(df, price_df[["abs_week", "-1w_change", "volume", "-1w_change_volume"]], on=["abs_week"], how="left")
        df.iloc[:,2:] = df.iloc[:,2:].fillna(0.) 

        # the df cannot be empty

        # creating y dataframe
        ydf = price_df[price_df["abs_week"].between(start_day.iloc[0], stop_day.iloc[0]-1)][["abs_week", "1w_change", "2w_change", "3w_change", "4w_change"]]
        assert len(ydf) == len(df), "dfs must be equal"
        assert not (ydf.isna().any().any()), "df is empty"
        
        if not self.to_save:
            X = torch.Tensor(df.drop(["tick", "abs_week"], axis=1).astype("float32").values)
            y = torch.Tensor(ydf.drop("abs_week", axis=1).astype("float32").values)
            return X, y
        else:
            return df.merge(ydf, on="abs_week", how="inner")




if __name__ == "__main__":

    start_time = None #datetime.datetime.strptime("02-01-2020", "%d-%m-%Y")
    end_time = None # datetime.datetime.strptime("01-03-2023", "%d-%m-%Y")
    dt = TradeDataset(1, start_date=start_time, end_date=end_time, to_save=True)

    times = []
    data = []
    data_present = 0
    dt[0].to_csv('dataset2.csv')

    for i in tqdm(range(1, len(dt))):
        dt[i].to_csv('dataset2.csv', mode='a', header=False)


    print("Saving!")
