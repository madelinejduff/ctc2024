import numpy as np
import pandas as pd
from scipy.stats import norm
from datetime import datetime

class Strategy:

    def __init__(self, risk_free_rate=0.03, volatility=0.2) -> None:
        self.capital: float = 1_000_000
        self.portfolio: float = 0
        self.current_delta: float = 0  # Track current portfolio delta for hedging
        self.risk_free_rate = risk_free_rate  # Annual risk-free interest rate
        self.volatility = volatility  # Assumed constant volatility for the underlying

        # Load options data
        self.options: pd.DataFrame = pd.read_csv(r"data/cleaned_options_data.zip")
        self.options["ts_clean"]    = pd.to_datetime(self.options["ts_recv"]).dt.tz_convert("US/Eastern")
        self.options["day"]         = self.options["ts_clean"].dt.date
        self.options["hour"]        = self.options["ts_clean"].dt.hour
        self.options["expiration"]  = pd.to_datetime(self.options.symbol.apply(lambda x: x[6:12]), format="%y%m%d").dt.date
        self.options["days_to_exp"] = (self.options["expiration"] - self.options["day"]).apply(lambda x: x.days)
        self.options["strike"]      = self.options.symbol.apply(lambda x: int(x[-8:])/1000.0)
        self.options["is_call"]     = self.options.symbol.apply(lambda x: x[-9] == "C")

        # Load underlying data
        self.underlying = pd.read_csv(r"data/underlying_data_hour.csv")
        self.underlying.columns = self.underlying.columns.str.lower()
        self.underlying["date"] = pd.to_datetime(self.underlying["date"]).dt.tz_convert("US/Eastern")

    def calculate_delta(self, option_row, underlying_price, current_date):
        """
        Calculate delta for each option using the Black-Scholes formula.
        """
        try:
            # Extract the strike price and expiration date
            strike_price = float(option_row["strike"])
            expiration_date = pd.Timestamp(self.extract_expiration_date(option_row["symbol"]))

            # Ensure current_date and expiration_date are timezone-naive
            current_date = current_date.tz_localize(None)
            expiration_date = expiration_date.tz_localize(None)

            T = (expiration_date - current_date).days / 365.0  # Time to expiration in years
            if T <= 0:
                return np.nan

            # Calculate delta using Black-Scholes
            if "C" in option_row["symbol"]:
                return self.black_scholes_delta(S=underlying_price, K=strike_price, T=T, r=self.risk_free_rate, sigma=self.volatility, option_type='C')
            elif "P" in option_row["symbol"]:
                return self.black_scholes_delta(S=underlying_price, K=strike_price, T=T, r=self.risk_free_rate, sigma=self.volatility, option_type='P')

        except Exception as e:
            print(f"Error in calculate_delta: {e}")
        return 0

    def generate_orders(self, underlying_prices) -> pd.DataFrame:
        """
        Generate orders based on delta-neutral strategy using exact timestamps for hedging.
        """
        orders = []
        underlying_prices["date"] = pd.to_datetime(underlying_prices["date"])

        for date, hour in underlying_prices[["date", "hour"]].drop_duplicates().values:
            available_options = self.options[
                (self.options["day"] == date.date()) &
                (self.options["hour"] == hour) &
                (self.options["days_to_exp"] > 14)
            ]

            if available_options.empty:
                continue

            call_options = available_options[available_options.is_call == True]
            put_options  = available_options[available_options.is_call == False]

            if call_options.empty or put_options.empty:
                continue

            price = underlying_prices.loc[
                (underlying_prices["date"] == date) &
                (underlying_prices["hour"] == hour), 
                "open"
            ].values[0]

            # Sort for ATM options
            sorted_calls = call_options.sort_values(by='strike', key=lambda x: abs(x - price))
            sorted_puts  = put_options.sort_values(by='strike', key=lambda x: abs(x - price))

            atm_call = sorted_calls.iloc[0]
            atm_put  = sorted_puts.iloc[0]

            # Calculate delta for ATM options
            call_delta = self.calculate_delta(atm_call, price, date)
            put_delta  = self.calculate_delta(atm_put, price, date)

            if np.isnan(call_delta) or np.isnan(put_delta):
                continue

            if self.current_delta == 0:
                self.current_delta = call_delta - put_delta

            if self.current_delta < 0:
                order_size = min(int(atm_call["ask_sz_00"]), 5)
                orders.append({
                    "datetime": atm_call["ts_recv"],
                    "option_symbol": atm_call["symbol"],
                    "action": "B",  
                    "order_size": order_size
                })
                self.current_delta += call_delta * order_size

            if self.current_delta > 0:
                order_size = min(int(atm_put["ask_sz_00"]), 5)
                orders.append({
                    "datetime": atm_put["ts_recv"],
                    "option_symbol": atm_put["symbol"],
                    "action": "B",  
                    "order_size": order_size
                })
                self.current_delta += put_delta * order_size

        return pd.DataFrame(orders)
