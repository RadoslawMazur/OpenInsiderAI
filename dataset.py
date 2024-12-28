from torch.utils.data import Dataset
import torch
from typing import List
import pandas as pd
from time import time_ns
import random
import isoweek
from functools import cache
import os


class TradeDataset(Dataset):

    def _get_world_gdp(self) -> pd.DataFrame:
        name = "NYGDPMKTPCDWLD"
        df = pd.read_csv(f"{name}.csv", parse_dates=[0])
        for i in range(1,6):
            df[f"{i}y_change"] = (df[name] / df[name].shift(i) - 1) * 100

        df[["year", "week", "day"]] = pd.to_datetime(df["DATE"]).dt.isocalendar()
        df["abs_week"] = df["week"] + df["year"].apply(self._add_weeks_per_year)
        df = df[df["abs_week"] > 0]
        df = pd.merge(pd.DataFrame(range(df["abs_week"].min(), df["abs_week"].max()), columns=["abs_week"]), df, how="left").bfill()

        return df[["abs_week", name, "1y_change", "2y_change", "3y_change", "4y_change", "5y_change"]]
    
    def _get_us_gdp(self) -> pd.DataFrame:
        name = "GDPC1"
        df = pd.read_csv(f"{name}.csv")
        for i in range(1,4):
            df[f"{i}q_change"] = (df[name] / df[name].shift(i) - 1) * 100
        for i in range(1*4,4*4, 4):
            df[f"{i}y_change"] = (df[name] / df[name].shift(i) - 1) * 100

        df[["year", "week", "day"]] = pd.to_datetime(df["DATE"]).dt.isocalendar()
        df["abs_week"] = df["week"] + df["year"].apply(self._add_weeks_per_year)
        df = df[df["abs_week"] > 0]
        df = pd.merge(pd.DataFrame(range(df["abs_week"].min(), df["abs_week"].max()), columns=["abs_week"]), df, how="left").bfill()
        
        return df[["abs_week", name, "1q_change", "2q_change", "3q_change", "4y_change", "8y_change", "12y_change"]]
    
    def _get_interest_rate(self) -> pd.DataFrame:
        name = "INTDSRUSM193N"
        df = pd.read_csv(f"{name}.csv")
        for i in range(3,15,3):
            df[f"{i}m"] = df[name].shift(i)
        
        df[["year", "week", "day"]] = pd.to_datetime(df["DATE"]).dt.isocalendar()
        df["abs_week"] = df["week"] + df["year"].apply(self._add_weeks_per_year)
        df = df[df["abs_week"] > 0]
        df = pd.merge(pd.DataFrame(range(df["abs_week"].min(), df["abs_week"].max()), columns=["abs_week"]), df, how="left").bfill()

        return df[["abs_week", name, "3m", "6m", "9m", "12m"]]

    @cache # cache is crucial here
    def _add_weeks_per_year(self, year) -> int:
        start_year = 1999
        weeks = 0 if year>=start_year else -999
        for y in range(start_year, year):
            weeks += isoweek.Week.last_week_of_year(y)[1]
        return weeks

    def _get_insider_trades(self) -> pd.DataFrame:
        df = pd.read_csv("oi_csv_2.csv", usecols=["Filling", "Trade", "tick", "title", "type", "price", "qty", "owned", "delta_owned", "value"], nrows=1_000_000).drop_duplicates().dropna(subset=["Filling"])

        # filter ticks to those price history is available
        # TODO get more ticks!
        df["tick"] = df["tick"].str.strip(" ")
        df = df[df["tick"].isin(map(lambda x: x[:-4], os.listdir("data")))]

        # remove ticks with less than 50 transactions (??)
        counts_df = df["tick"].value_counts()
        counts_df = counts_df[counts_df > 50] 
        df = df[df["tick"].isin(counts_df.index)]

        # parse the numbers
        df["price"] = df["price"].str.replace("[$,]", "", regex=True).astype("float32")
        df["qty"] = df["qty"].str.replace(",", "").fillna(0).astype("int32")
        df["owned"] = df["owned"].str.replace(",", "").fillna(0).astype("int32")
        df["delta_owned"] = df["delta_owned"].str.rstrip("%").str.replace("New", "0").str.replace(">999", "1000").fillna(0).astype("int16")
        df["value"] = df["value"].str.replace("[$,]", "", regex=True).fillna(0).astype("int32")

        # parse the dates
        df["Trade"] =  pd.to_datetime(df["Trade"])
        df["Filling"] =  pd.to_datetime(df["Filling"])

        # get some info from title
        df["is_dir"] = df["title"].str.lower().str.contains("dir")
        df["is_ceo"] = df["title"].str.lower().str.contains("ceo")
        df["is_major_steakholder"] = df["title"].str.lower().str.contains("10%")
        df = df.drop("title", axis=1)

        # days to filling
        df["d_to_filling"] = (df["Trade"] - df["Filling"]).dt.days
        df = df.drop("Filling", axis=1)

        # split trade date
        df[["t_year", "t_week", "t_day"]] = df["Trade"].dt.isocalendar()
        df["abs_week"] = df["t_week"] + df["t_year"].apply(self._add_weeks_per_year)
        df = df.drop(["Trade"], axis=1)

        # one hot encoding of transaction type
        transaction_types = pd.get_dummies(df["type"]) * 1
        df = pd.merge(df, transaction_types, left_index=True, right_index=True)
        df = df.drop("type", axis=1)

        return df.groupby(["tick", "abs_week"]).mean().reset_index()
    
    def _get_tick_borders(self):

        # this is to remove rows we don't have data on #FIXME
        df = self.insider_df.loc[self.insider_df["abs_week"] < 1090]
        # this is a function to get first and last insider trade for each tick
        df = pd.merge(df.groupby("tick")["abs_week"].max() - self.seq_len, df.groupby("tick")["abs_week"].min(), left_index=True, right_index=True)

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
        df = pd.read_csv(f"data/{tick}.csv")

        # get weeks from 1999
        df["Date"] = pd.to_datetime(df["Date"])
        df[["year", "week", "day"]] = df["Date"].dt.isocalendar()
        df["abs_week"] = df["week"] + df["year"].apply(self._add_weeks_per_year)
        df["mean_price"] = (df["High"] + df["Low"])/2

        # figure out mean week price
        df = df[["abs_week", "year", "week", "mean_price"]].groupby(["year", "week"]).mean().reset_index()

        # calcaulte percent change in price from of one week to other
        for i in range(1,5):
            df[f"{i}w_change"] = ((df["mean_price"] / df["mean_price"].shift(i)) - 1) * 100

        df = df[df["abs_week"] > 0]

        return df[["abs_week", "1w_change", "2w_change", "3w_change", "4w_change"]]

    def __init__(self, seq_len: int):
        self.seq_len = seq_len
        self.world_gdp_df = self._get_world_gdp()
        self.us_gdp_df = self._get_us_gdp()
        self.interest_rate = self._get_interest_rate()
        self.insider_df = self._get_insider_trades()
        self.tick_borders = self._get_tick_borders()

    def __len__(self) -> int:
        #return (self.insider_df.groupby("tick")["abs_week"].max() - self.insider_df.groupby("tick")["abs_week"].min() - self.seq_len).sum()
        return self.tick_borders["n_weeks"].sum()

    def __getitem__(self, index: int) -> List[torch.Tensor]:

        # select which the tick using rolling week as an index
        week = self.tick_borders.loc[self.tick_borders["rolling_week"] < index, "rolling_week"].max()
        tick = self.tick_borders.loc[(self.tick_borders["rolling_week"] == week), "tick"]

        # 
        start_day = self.tick_borders.loc[self.tick_borders["rolling_week"] == week, "abs_week_y"] + index - week
        stop_day = start_day + self.seq_len

        # assetions to confirm 
        assert start_day.size != 0 or stop_day.size != 0 or start_day.size > 2 or stop_day.size > 2, "there should be only one row"
        assert tick.size != 0 or tick.size > 2, "Only one tick should be selected"

        # creating X dataframe
        df = pd.DataFrame(zip([tick.iloc[0]] * self.seq_len, range(start_day.iloc[0], stop_day.iloc[0])), columns=["tick", "abs_week"])
        df = pd.merge(df, self.insider_df, on=["tick", "abs_week"], how="left").fillna(0)
        df = pd.merge(df, self.interest_rate, on=["abs_week"], how="left") #TODO premerge those dfs
        df = pd.merge(df, self.us_gdp_df, on=["abs_week"], how="left")
        df = pd.merge(df, self.world_gdp_df, on=["abs_week"], how="left")

        # the df cannot be empty
        assert ~(df.isna().any().any()), "df is empty"

        # creating y dataframe
        ydf = self._get_tick_dataframe(tick.iloc[0])
        ydf = ydf[ydf["abs_week"].between(start_day.iloc[0], stop_day.iloc[0]-1)][["abs_week","1w_change", "2w_change", "3w_change", "4w_change"]]
        assert len(ydf) == len(df), "dfs must be equal"

        X = torch.Tensor(df.drop(["tick", "abs_week"], axis=1).astype("float32").values)
        y = torch.Tensor(ydf.drop("abs_week", axis=1).astype("float32").values)

        return X, y


if __name__ == "__main__":
    dt = TradeDataset(8)
    times = []
    data = []
    print(len(dt))
    for _ in range(1000):
        i = random.choice(range(0, len(dt)))
        start_time = time_ns()
        data.append((dt[i][0][:,2:10] > 10).sum() > 0)
        times.append((time_ns() - start_time)/10**6)
    
    print("Average execution time:")
    print(sum(times)/len(times))
    print("ms")

    print(sum(data)/len(data))

    pass