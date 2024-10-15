import numpy as np
import pandas as pd
from scipy.stats import norm
from datetime import datetime, timedelta

class Strategy:
    def __init__(self, risk_free_rate=0.03, volatility=0.2) -> None:
        self.capital: float = 1_000_000
        self.portfolio: float = 0
        self.current_delta: float = 0  # Track current portfolio delta for hedging
        self.risk_free_rate = risk_free_rate  # Annual risk-free interest rate
        self.volatility = volatility  # Assumed constant volatility for the underlying
        self.transaction_cost = 0.50  # Per contract transaction cost
        self.slippage = 0.001  # Slippage is 0.1%

        # Load options data
        self.options: pd.DataFrame = pd.read_csv(r"data/cleaned_options_data.zip")
        
        # Convert to UTC timezone
        self.options["ts_clean"] = pd.to_datetime(self.options["ts_recv"]).dt.tz_convert("UTC")
        self.options["day"] = self.options["ts_clean"].dt.date
        self.options["hour"] = self.options["ts_clean"].dt.hour
        self.options["expiration"] = pd.to_datetime(self.options.symbol.apply(lambda x: x[6:12]), format="%y%m%d").dt.date
        self.options["days_to_exp"] = (self.options["expiration"] - self.options["day"]).apply(lambda x: x.days)
        self.options["strike"] = self.options.symbol.apply(lambda x: int(x[-8:])/1000.0)
        self.options["is_call"] = self.options.symbol.apply(lambda x: x[-9] == "C")

        # Load underlying data
        self.underlying = pd.read_csv(r"data/underlying_data_hour.csv")
        self.underlying.columns = self.underlying.columns.str.lower()

        # Ensure the 'date' column is in UTC
        self.underlying["date"] = pd.to_datetime(self.underlying["date"], utc=True).dt.tz_convert("UTC")

        # Floor the 'date' column to the hour and extract day and hour
        self.underlying["date"] = self.underlying["date"].dt.floor("H")
        self.underlying["day"] = self.underlying["date"].dt.date
        self.underlying["hour"] = self.underlying["date"].dt.hour

        # Calculate the 12-hour moving average
        self.underlying['moving_average'] = self.underlying['open'].rolling(window=48).mean()

    def black_scholes_delta(self, S, K, T, r, sigma, option_type='C'):
        """
        Calculate Black-Scholes delta for call or put option.
        """
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        if option_type == 'C':
            return norm.cdf(d1)  # Call delta
        elif option_type == 'P':
            return norm.cdf(d1) - 1  # Put delta

    def generate_orders(self) -> pd.DataFrame:
        """
        Generate market-neutral orders based on the moving average.
        Buy or sell both calls and puts within a range of strike prices around the ATM.
        Hedge the portfolio delta after each trade to maintain delta neutrality.
        """
        orders = []
        num_atm_options = 3  # Define how many closest ATM options to consider
        min_date = self.underlying["day"].min()
        skip_days_threshold = pd.Timestamp(min_date) + pd.Timedelta(days=1)

        for idx, row in self.underlying.iterrows():
            current_date = pd.Timestamp(row["day"])
            if current_date < skip_days_threshold:
                continue

            current_price = row['open']
            ma_price = row['moving_average']
            current_day = row['day']
            current_hour = row['hour']

            available_options = self.options[
                (self.options["day"] == current_day) &
                (self.options["hour"] == current_hour) &
                (self.options["days_to_exp"] < 5)  # Options with less than 5 days to expiration
            ]

            if available_options.empty:
                continue

            available_options_last_hour = available_options.copy()
            available_options_last_hour['strike_diff'] = (available_options_last_hour['strike'] - current_price).abs()

            closest_calls = available_options_last_hour[available_options_last_hour['is_call']].sort_values(by='strike_diff').head(num_atm_options)
            closest_puts = available_options_last_hour[~available_options_last_hour['is_call']].sort_values(by='strike_diff').head(num_atm_options)

            if closest_calls.empty or closest_puts.empty:
                continue

            # Generate orders based on price vs. moving average
            for _, call_option in closest_calls.iterrows():
                for _, put_option in closest_puts.iterrows():
                    if pd.Timestamp(call_option['expiration']) <= current_date + timedelta(days=1) or \
                       pd.Timestamp(put_option['expiration']) <= current_date + timedelta(days=1):
                        continue

                    # Calculate the delta for both call and put options
                    T_call = (call_option['expiration'] - current_date).days / 365
                    T_put = (put_option['expiration'] - current_date).days / 365
                    call_delta = self.black_scholes_delta(current_price, call_option['strike'], T_call, self.risk_free_rate, self.volatility, 'C')
                    put_delta = self.black_scholes_delta(current_price, put_option['strike'], T_put, self.risk_free_rate, self.volatility, 'P')

                    if current_price > ma_price:
                        # Uptrend: Buy call and put, hedge with underlying
                        self.place_order(orders, call_option, 'B', call_delta)
                        self.place_order(orders, put_option, 'B', put_delta)
                    elif current_price < ma_price:
                        # Downtrend: Sell call and put, hedge with underlying
                        self.place_order(orders, call_option, 'S', call_delta)
                        self.place_order(orders, put_option, 'S', put_delta)

                    # Hedge portfolio delta
                    self.hedge_portfolio(call_delta, put_delta)

        orders_df = pd.DataFrame(orders)
        orders_df = orders_df.groupby(orders_df['datetime'].dt.date).apply(lambda x: x.head(20)).reset_index(drop=True)
        return orders_df

    def place_order(self, orders, option, action, delta):
        order_size = min(int(option['ask_sz_00']), 5)  # Example logic for order size
        orders.append({
            'datetime': option['ts_clean'],
            'option_symbol': option['symbol'],
            'action': action,
            'order_size': order_size,
            'delta': delta
        })

    def hedge_portfolio(self, call_delta, put_delta):
        """
        Adjust portfolio delta to maintain delta neutrality.
        If current delta is positive, hedge by buying puts (negative delta).
        If current delta is negative, hedge by buying calls (positive delta).
        """
        net_delta = self.current_delta + call_delta - put_delta
        if net_delta > 0:
            # Hedge with puts
            print(f"Hedging with puts to offset delta of {net_delta}")
        elif net_delta < 0:
            # Hedge with calls
            print(f"Hedging with calls to offset delta of {net_delta}")
        self.current_delta = net_delta
