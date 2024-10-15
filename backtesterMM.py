import numpy as np
import pandas as pd
from typing import List
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

class Backtester:

    def __init__(self, start_date, end_date, strategy) -> None:
        self.capital: float = 100_000_000
        self.portfolio_value: float = 0

        self.start_date: datetime = start_date
        self.end_date: datetime = end_date

        self.user_strategy = strategy

        # Load options and underlying data
        self.options: pd.DataFrame = pd.read_csv("data/cleaned_options_data.zip")
        self.options["day"] = self.options["ts_recv"].apply(lambda x: x.split("T")[0])
        self.options["hour"] = self.options["ts_recv"].apply(lambda x: int(x.split("T")[1].split(".")[0].split(":")[0]))
        self.options["ts_recv"] = pd.to_datetime(self.options["ts_recv"], utc=True)

        self.underlying = pd.read_csv(r"data/underlying_data_hour.csv", header=0)
        self.underlying.columns = self.underlying.columns.str.lower()
        self.underlying["hour"] = self.underlying["date"].apply(lambda x: int(x.split(" ")[1].split("-")[0].split(":")[0]))
        self.underlying["date"] = pd.to_datetime(self.underlying["date"], utc=True)
        self.underlying["day"] = pd.to_datetime(self.underlying["date"]).dt.date

        self.orders: pd.DataFrame = self.user_strategy.generate_orders()
        self.orders["day"] = self.orders["datetime"].dt.date
        self.orders["hour"] = self.orders["datetime"].dt.hour
        self.orders["expiration_date"] = self.orders["option_symbol"].apply(lambda x: self.get_expiration_date(x))
        self.orders["sort_by"] = pd.to_datetime(self.orders["datetime"])
        self.orders = self.orders.sort_values(by="sort_by")

        self.pnl: List = []
        self.max_drawdown: float = float("-inf")
        self.overall_return: float = 0
        self.sharpe_ratio: float = 0
        self.overall_score: float = 0
        self.open_orders: pd.DataFrame = pd.DataFrame(columns=["day", "datetime", "option_symbol", "action", "order_size", "expiration_date", "hour"])
        self.open_orders["order_size"] = self.open_orders["order_size"].astype(float)

    def get_expiration_date(self, symbol) -> str:
        numbers: str = symbol.split(" ")[3]
        date: str = numbers[:6]
        date_yymmdd: str = "20" + date[0:2] + "-" + date[2:4] + "-" + date[4:6]
        return date_yymmdd

    def parse_option_symbol(self, symbol) -> List:
        """
        example: SPX   240419C00800000
        """
        numbers: str = symbol.split(" ")[3]
        date: str = numbers[:6]
        date_yymmdd: str = "20" + date[0:2] + "-" + date[2:4] + "-" + date[4:6]
        action: str = numbers[6]
        strike_price: float = float(numbers[7:]) / 1000
        return [datetime.strptime(date_yymmdd, "%Y-%m-%d"), action, strike_price]

    def check_option_is_open(self, row: pd.Series) -> bool:
        same: pd.DataFrame = self.open_orders[
            (self.open_orders["option_symbol"] == row["option_symbol"]) &
            (self.open_orders["datetime"] == row["datetime"])
        ]
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
        delta: timedelta = timedelta(days=1)
        current_date: datetime = self.start_date

        while current_date <= self.end_date:
            print(f"\n--- Processing date: {current_date} ---")
            for _, row in self.orders.iterrows():
                if str(current_date).split(" ")[0] == str(row["day"]):
                    option_metadata: List = self.parse_option_symbol(row["option_symbol"])
                    order_size: float = float(row["order_size"])
                    strike_price: float = option_metadata[2]

                    matching_row = self.options[
                        (self.options["symbol"] == row["option_symbol"]) & 
                        (self.options["ts_recv"] == row["datetime"])
                    ]

                    if not matching_row.empty:
                        matching_row = matching_row.iloc[0]
                    else:
                        print(f"** No matching row found for {row['option_symbol']} at {row['datetime']} **")
                        continue

                    ask_price = float(matching_row["ask_px_00"])
                    buy_price = float(matching_row["bid_px_00"])
                    ask_size = float(matching_row["ask_sz_00"])
                    buy_size = float(matching_row["bid_sz_00"])

                    # Ensure that order size does not exceed available liquidity
                    if order_size < 0:
                        raise ValueError("Order size must be positive")

                    if (row["action"] == "B" and order_size > ask_size) or (row["action"] == "S" and order_size > buy_size):
                        raise ValueError(f"Order size exceeds available size; order size: {order_size}, ask size: {ask_size}, buy size: {buy_size}; action: {row['action']}")

                    # Handle buy logic
                    if row["action"] == "B":
                        options_cost: float = order_size * ask_price + 0.1 * strike_price
                        transaction_cost: float = 0.5 * order_size
                        slippage_cost: float = options_cost * 0.001  # 0.1% slippage

                        print(f"Buying {order_size} of {row['option_symbol']} at {ask_price}")
                        print(f"Transaction cost: {transaction_cost}, Slippage cost: {slippage_cost}")

                        if self.capital >= options_cost + transaction_cost + slippage_cost:
                            self.capital -= options_cost + transaction_cost + slippage_cost
                            self.portfolio_value += order_size * ask_price
                            print(f"Capital after buying: {self.capital}, Portfolio value: {self.portfolio_value}")
                            if not self.check_option_is_open(row):
                                self.open_orders.loc[len(self.open_orders)] = row
                        else:
                            print(f"Not enough capital for trade: Required: {options_cost + transaction_cost + slippage_cost}, Available: {self.capital}")

                    # Handle sell logic
                    else:
                        row["hour"] = min(row["hour"], 15)
                        matching_underlying_rows = self.underlying[
                            (self.underlying["day"] == row["day"]) & 
                            (self.underlying["hour"] == row["hour"])
                        ]

                        if len(matching_underlying_rows) != 1:
                            print(f"Skipping due to missing or duplicate data for day: {row['day']} and hour: {row['hour']}")
                            continue

                        underlying_price: float = float(matching_underlying_rows["adj close"].iloc[0])
                        sold_stock_cost: float = order_size * 100 * underlying_price
                        open_price: float = float(matching_underlying_rows["open"].iloc[0])
                        margin: float = 100 * order_size * (buy_price + 0.1 * open_price)

                        if (self.capital + order_size * buy_price + 0.1 * strike_price) > margin and (self.capital + order_size * buy_price + 0.1 * strike_price - sold_stock_cost + 0.5) > 0:
                            self.capital += order_size * buy_price
                            self.capital -= sold_stock_cost + 0.5
                            self.portfolio_value += order_size * 100 * underlying_price
                            print(f"Selling option {row['option_symbol']} for {buy_price}")
                            if not self.check_option_is_open(row):
                                self.open_orders.loc[len(self.open_orders)] = row

            print(f"End of day: {current_date}, Capital: {self.capital}, Portfolio Value: {self.portfolio_value}")

            # Process expired orders
            for _, order in self.open_orders.iterrows():
                option_metadata: List = self.parse_option_symbol(order["option_symbol"])
                if str(order["expiration_date"]) == str(current_date).split(" ")[0]:
                    print(f"Processing expiration of {order['option_symbol']} on {current_date}")
                    order["hour"] = min(order["hour"], 15)
                    matching_underlying_rows = self.underlying[
                        (self.underlying["day"] == order["day"]) & 
                        (self.underlying["hour"] == order["hour"])
                    ]

                    if len(matching_underlying_rows) != 1:
                        print(f"Skipping due to missing or duplicate data for day: {order['day']} and hour: {order['hour']}")
                        continue

                    underlying_price: float = float(matching_underlying_rows["adj close"].iloc[0])
                    put_call: str = option_metadata[1]
                    strike_price: float = option_metadata[2]
                    order_size: float = float(order["order_size"])
                    underlying_cost: float = strike_price * 100 * order_size

                    if order["action"] == "B":
                        if put_call == "C":
                            if underlying_price > strike_price:
                                profit = 100 * order_size * (underlying_price - strike_price)
                                print(f"Call expired in the money: {order['option_symbol']}, Profit: {profit}")
                                self.capital += profit
                                self.portfolio_value -= underlying_cost
                        else:
                            if underlying_price < strike_price:
                                print(f"Put expired in the money: {order['option_symbol']}, Profit: {underlying_cost}")
                                self.capital += underlying_cost
                    else:
                        if put_call == "C":
                            if underlying_price > strike_price:
                                loss = order_size * 100 * (underlying_price - strike_price)
                                print(f"Call sold at loss: {order['option_symbol']}, Loss: {loss}")
                                self.portfolio_value -= loss
                        else:
                            if underlying_price < strike_price:
                                cost = order_size * 100 * (strike_price - underlying_price)
                                print(f"Put sold at profit: {order['option_symbol']}, Profit: {cost}")
                                self.capital -= cost
                                self.portfolio_value += cost

            self.portfolio_value = max(self.portfolio_value, 0)
            self.open_orders = self.open_orders[self.open_orders["expiration_date"] != str(current_date).split(" ")[0]]
            current_date += delta
            self.pnl.append(self.capital + self.portfolio_value)

            print(f"Date: {current_date}, Capital: {self.capital}, Portfolio Value: {self.portfolio_value}, Total PnL: {self.pnl[-1]}")

        # Handle open orders past expiration
        for _, order in self.open_orders.iterrows():
            option_metadata: List = self.parse_option_symbol(order["option_symbol"])
            last_row: pd.Series = self.underlying.iloc[-1]
            print(f"Closing open order: {order['option_symbol']}")
            if (option_metadata[1] == "B"):
                self.portfolio_value -= last_row["adj close"] * 100 * order["order_size"]
                self.capital += 0.9 * (last_row["adj close"] * 100 * order["order_size"])
            else:
                self.portfolio_value += last_row["adj close"] * 100 * order["order_size"]
                self.capital -= 1.1 * (last_row["adj close"] * 100 * order["order_size"])

        self.pnl.append(self.capital + self.portfolio_value)
        print(f"After closing open orders: final capital: {self.capital}, final portfolio value: {self.portfolio_value}, final pnl: {self.pnl[-1]}")

    def compute_overall_score(self):
        ptr: int = 0
        high_point: float = float("-inf")
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
