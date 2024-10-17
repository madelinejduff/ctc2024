import numpy as np
import pandas as pd
from typing import List
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

class Backtester:

    def __init__(self, start_date, end_date, strategy) -> None:
        self.capital : float = 100_000_000
        self.portfolio_value : float = 0

        self.start_date : datetime = start_date
        self.end_date : datetime = end_date

        self.user_strategy = strategy

        self.options : pd.DataFrame = pd.read_csv("data/cleaned_options_data.zip")
        self.options["day"] = self.options["ts_recv"].apply(lambda x: x.split("T")[0])
        self.options["hour"] = self.options["ts_recv"].apply(lambda x: int(x.split("T")[1].split(".")[0].split(":")[0]))

        self.underlying = pd.read_csv(r"data/underlying_data_hour.csv", header=0)
        self.underlying.columns = self.underlying.columns.str.lower()
        self.underlying["hour"] = self.underlying["date"].apply(lambda x : int(x.split(" ")[1].split("-")[0].split(":")[0]))
        self.underlying["date"] = pd.to_datetime(self.underlying["date"], utc=True)
        self.underlying["day"] = pd.to_datetime(self.underlying["date"]).dt.date

        self.orders : pd.DataFrame = self.user_strategy.generate_orders()
#             self.underlying
        self.orders["day"] = self.orders["datetime"].dt.date
        self.orders["hour"] = self.orders["datetime"].dt.hour
#         self.orders["day"] = self.orders["datetime"].apply(lambda x: x.split("T")[0])
#         self.orders["hour"] = self.orders["datetime"].apply(lambda x: int(x.split("T")[1].split(".")[0].split(":")[0]))
        self.orders["expiration_date"] = self.orders["option_symbol"].apply(lambda x: self.get_expiration_date(x))
        self.orders["sort_by"] = pd.to_datetime(self.orders["datetime"])
        self.orders = self.orders.sort_values(by="sort_by")

        self.pnl : List = []
        self.max_drawdown : float = float("-inf")
        self.overall_return : float = 0
        self.sharpe_ratio : float = 0
        self.overall_score : float = 0
        self.open_orders : pd.DataFrame = pd.DataFrame(columns=["day", "datetime", "option_symbol", "action", "order_size", "expiration_date", "hour"])
        self.open_orders["order_size"] = self.open_orders["order_size"].astype(float)

    def get_expiration_date(self, symbol) -> str:
        numbers : str = symbol.split(" ")[3]
        date : str = numbers[:6]
        date_yymmdd : str = "20" + date[0:2] + "-" + date[2:4] + "-" + date[4:6]
        return date_yymmdd

    def parse_option_symbol(self, symbol) -> List:
        """
        example: SPX   240419C00800000
        """
        numbers : str = symbol.split(" ")[3]
        date : str = numbers[:6]
        date_yymmdd : str = "20" + date[0:2] + "-" + date[2:4] + "-" + date[4:6]
        action : str = numbers[6]
        strike_price : float = float(numbers[7:]) / 1000
        return [datetime.strptime(date_yymmdd, "%Y-%m-%d"), action, strike_price]
  
    def check_option_is_open(self, row: pd.Series) -> bool:
        same: pd.DataFrame = self.open_orders[(self.open_orders["option_symbol"] == row["option_symbol"]) 
                                              & (self.open_orders["datetime"] == row["datetime"])]
        if len(same) > 0:
          assert len(same) == 1
          assert float(row["order_size"])
          same_index: int = same.index[0]
          if row["action"] == same["action"].iloc[0]:
            self.open_orders.loc[same_index, "order_size"] += float(row["order_size"])
          else:
            if row["order_size"] > same["order_size"].iloc[0]:
              self.open_orders.loc[same_index, "action"] = "B" if row["action"] == "S" else "S"
              self.open_orders.loc[same_index, "order_size"] = float(row["order_size"] - same["order_size"].iloc[0])
            elif row["order_size"] == same["order_size"].iloc[0]:
              self.open_orders = self.open_orders.drop(index=same_index)
            else:
              self.open_orders.loc[same_index, "order_size"] -= float(row["order_size"])
          return True
        return False
    
    def calculate_pnl(self):
        """
        Calculate PnL for each day by processing executed orders and updating the portfolio value.
        """
        delta: timedelta = timedelta(days=1)
        current_date: datetime = self.start_date

        # Ensure orders and options datetime columns are timezone-aware and aligned
        self.orders["datetime"] = pd.to_datetime(self.orders["datetime"], utc=True).dt.tz_convert("US/Eastern")
        self.options["ts_recv"] = pd.to_datetime(self.options["ts_recv"], utc=True).dt.tz_convert("US/Eastern")

        # Floor datetime precision to the nearest second or minute to avoid nanosecond precision mismatches
        self.orders["datetime"] = self.orders["datetime"].dt.floor("T")  # Or use 'S' for seconds
        self.options["ts_recv"] = self.options["ts_recv"].dt.floor("T")  # Or use 'S' for seconds

        while current_date <= self.end_date:
            # Process orders for the current day
            for _, row in self.orders.iterrows():
                if str(current_date).split(" ")[0] == str(row["day"]):
                    option_metadata: List = self.parse_option_symbol(row["option_symbol"])
                    order_size: float = float(row["order_size"])
                    strike_price: float = option_metadata[2]

                    # Match the order with the corresponding option in the dataset using floored timestamps
                    matching_row = self.options[
                        (self.options["symbol"] == row["option_symbol"]) &
                        (self.options["ts_recv"] == row["datetime"])
                    ]

                    if matching_row.empty:
                        print(f"No match found for option: {row['option_symbol']} at {row['datetime']}")
                        continue

                    matching_row = matching_row.iloc[0]
                    ask_price = float(matching_row["ask_px_00"])
                    bid_price = float(matching_row["bid_px_00"])
                    ask_size = float(matching_row["ask_sz_00"])
                    bid_size = float(matching_row["bid_sz_00"])

                    # Ensure order size is valid
                    if order_size <= 0:
                        print(f"Invalid order size: {order_size}")
                        continue

                    if row["action"] == "B" and order_size > ask_size:
                        print(f"Buy order size exceeds ask size for {row['option_symbol']}. Order size: {order_size}, Ask size: {ask_size}")
                        continue
                    elif row["action"] == "S" and order_size > bid_size:
                        print(f"Sell order size exceeds bid size for {row['option_symbol']}. Order size: {order_size}, Bid size: {bid_size}")
                        continue

                    # Check capital and margin conditions
                    margin = (ask_price + 0.1 * strike_price) * order_size
                    if self.capital < margin:
                        print(f"Insufficient capital for order {row['option_symbol']}. Required margin: {margin}, Available capital: {self.capital}")
                        continue

                    # Execute the order
                    if row["action"] == "B":
                        options_cost: float = order_size * ask_price + 0.1 * strike_price
                        self.capital -= options_cost + 0.5
                        self.portfolio_value += order_size * ask_price
                        print(f"Executed buy order for {row['option_symbol']} at price {ask_price}, order size {order_size}")

                    elif row["action"] == "S":
                        options_cost: float = order_size * bid_price
                        self.capital += options_cost
                        print(f"Executed sell order for {row['option_symbol']} at price {bid_price}, order size {order_size}")

            # Process open orders and check expiration
            for _, order in self.open_orders.iterrows():
                if str(order["expiration_date"]) == str(current_date).split(" ")[0]:
                    print(f"Order expired for {order['option_symbol']} on {current_date}")
                    self.open_orders = self.open_orders[self.open_orders["expiration_date"] != str(current_date).split(" ")[0]]

            # Update the PnL and advance to the next day
            self.pnl.append(self.capital + self.portfolio_value)
            print(f"Date: {current_date}, Capital: {self.capital}, Portfolio Value: {self.portfolio_value}, Total PnL: {self.pnl[-1]}")
            current_date += delta

        print(f"Final capital: {self.capital}, Final portfolio value: {self.portfolio_value}, Final PnL: {self.pnl[-1]}")

    def compute_overall_score(self):
        ptr : int = 0
        high_point : float = float("-inf")
        self.max_drawdown = 0.0

        while ptr < len(self.pnl):
          if self.pnl[ptr] > high_point:
            high_point = self.pnl[ptr]
          if self.pnl[ptr] < high_point:
            self.max_drawdown = max(self.max_drawdown, (high_point - self.pnl[ptr]) / high_point)
          ptr += 1

        print(f"Max Drawdown: {self.max_drawdown}")

        self.overall_return = 100 * ((self.pnl[-1] - 100_000_000) / 100_000_000)
        print(f"Overall Return: {self.overall_return}%")

        percentage_returns = []
        prev = 100_000_000
        for i in range(len(self.pnl)):
          percentage_returns.append(self.pnl[i] / prev)
          prev = self.pnl[i]

        avg_return = np.mean(percentage_returns)
        std_return = np.std(percentage_returns)

        if std_return > 0.0:
          risk_free_rate = 0.03 / 252
          self.sharpe_ratio = (avg_return - 1 - risk_free_rate) / std_return
          print(f"Sharpe Ratio: {self.sharpe_ratio}")
        else:
          self.sharpe_ratio = 0.0
          print("Sharpe Ratio: Undefined (Standard Deviation = 0)")

        if self.max_drawdown > 0 and self.sharpe_ratio > 0:
          self.overall_score = (self.overall_return / self.max_drawdown) * self.sharpe_ratio
          print(f"Overall Score: {self.overall_score}")
        else:
          print("Cannot calculate overall score (Max Drawdown or Sharpe Ratio <= 0)")

    def plot_pnl(self):
        if not isinstance(self.pnl, list) or len(self.pnl) == 0:
          print("PNL data is not available or empty")

        plt.figure(figsize=(10, 6))
        plt.plot(self.pnl, label="PnL", color="blue", marker="o", linestyle="-", markersize=5)

        plt.title("Profit and Loss Over Time", fontsize=14)
        plt.xlabel("Time (Days)", fontsize=12)
        plt.ylabel("PnL (Profit/Loss)", fontsize=12)

        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(loc="best")

        plt.tight_layout()
        plt.show()
