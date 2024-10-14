import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize
from datetime import datetime

class Strategy:

    def __init__(self, risk_free_rate=0.03, volatility=0.2) -> None:
        self.capital: float = 1_000_000
        self.portfolio: float = 0
        self.current_delta: float = 0  # Track current portfolio delta for hedging
        self.risk_free_rate = risk_free_rate  # Annual risk-free interest rate
        self.volatility = volatility  # Assumed constant volatility for the underlying

        # Load options data
        self.options: pd.DataFrame = pd.read_csv(r"data/cleaned_options_data.csv")
        self.options["day"] = pd.to_datetime(self.options["ts_recv"]).dt.date
        self.options["hour"] = pd.to_datetime(self.options["ts_recv"]).dt.hour  # Adding hour to filter by day and hour

        # Load underlying data
        self.underlying = pd.read_csv(r"data/underlying_data_hour.csv")
        self.underlying.columns = self.underlying.columns.str.lower()  # Standardize column names

    def black_scholes_delta(self, S, K, T, r, sigma, option_type='C'):
        """
        Calculate Black-Scholes delta for call or put option.
        """
        try:
            d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            if option_type == 'C':
                return norm.cdf(d1)  # Call delta
            elif option_type == 'P':
                return norm.cdf(d1) - 1  # Put delta
        except Exception as e:
            print(f"Error calculating Black-Scholes delta: {e}")
            return 0

    def black_scholes_price(self, S, K, T, r, sigma, option_type='C'):
        """
        Calculate Black-Scholes option price for a call or put.
        """
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if option_type == 'C':
            return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        elif option_type == 'P':
            return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    def backsolve_strike(self, S, option_price, T, r, sigma, option_type='C'):
        """
        Numerically solve for the strike price that matches the observed option price.
        """
        def objective(K):
            return abs(self.black_scholes_price(S, K, T, r, sigma, option_type) - option_price)

        # Initial guess for the strike price (let's assume it's near the current price)
        initial_guess = S
        result = minimize(objective, initial_guess, bounds=[(S*0.5, S*1.5)])  # Search within 50% to 150% of underlying price
        
        return result.x[0]  # Return the strike price

    def extract_expiration_date(self, option_symbol):
        """
        Extract the expiration date from the option symbol.
        Example: "SPX 240119C04700000" -> Expiration date is 2024-01-19.
        """
        try:
            # Assume the expiration date is in the format YYMMDD (e.g., "240119")
            expiration_str = option_symbol.split()[1][:6]  # Extract YYMMDD part
            expiration_date = datetime.strptime(expiration_str, "%y%m%d").date()
            return expiration_date
        except (ValueError, IndexError):
            print(f"Error extracting expiration date from symbol: {option_symbol}")
            return None  # Return None if there's an error
        
    def calculate_delta(self, option_row, underlying_price, current_date):
        """
        Calculate delta for each option using the Black-Scholes formula.
        If the strike price is missing, attempt to estimate it using backsolving.
        """
        try:
            # Ensure current_date is timezone-naive
            if current_date.tzinfo is not None:
                current_date = current_date.tz_localize(None)

            # If strike price is not available, backsolve it using the option price
            if "strike" not in option_row or pd.isna(option_row["strike"]):
                option_price = (option_row["ask_px_00"] + option_row["bid_px_00"]) / 2  # Use mid-price
                expiration_date = self.extract_expiration_date(option_row["symbol"])
                expiration_date = pd.Timestamp(expiration_date)  # Convert expiration_date to Timestamp

                # Make sure expiration_date is timezone-naive
                if expiration_date.tzinfo is not None:
                    expiration_date = expiration_date.tz_localize(None)

                T = (expiration_date - current_date).days / 365.0  # Time to expiration in years

                if T <= 0:
                    print(f"Skipped - Time to expiration (T) is zero or negative for symbol {option_row['symbol']}")
                    return np.nan

                option_type = 'C' if 'C' in option_row["symbol"] else 'P'

                strike_price = self.backsolve_strike(underlying_price, option_price, T, self.risk_free_rate, self.volatility, option_type)
            else:
                strike_price = float(option_row["strike"])

            expiration_date = self.extract_expiration_date(option_row["symbol"])
            expiration_date = pd.Timestamp(expiration_date)  # Convert expiration_date to Timestamp

            # Make sure expiration_date is timezone-naive
            if expiration_date.tzinfo is not None:
                expiration_date = expiration_date.tz_localize(None)

            T = (expiration_date - current_date).days / 365.0  # Time to expiration in years

            if T <= 0:
                print(f"Skipped - Time to expiration (T) is zero or negative for symbol {option_row['symbol']}")
                return np.nan

            # Use Black-Scholes to calculate delta
            if "C" in option_row["symbol"]:
                return self.black_scholes_delta(S=underlying_price, K=strike_price, T=T, r=self.risk_free_rate, sigma=self.volatility, option_type='C')
            elif "P" in option_row["symbol"]:
                return self.black_scholes_delta(S=underlying_price, K=strike_price, T=T, r=self.risk_free_rate, sigma=self.volatility, option_type='P')

        except Exception as e:
            print(f"Error in calculate_delta: {e}")
        return 0

    def generate_orders(self, underlying_prices) -> pd.DataFrame:
        """
        Generate orders based on delta-neutral strategy in a vectorized manner.
        Orders will be placed when the delta imbalance requires it.
        """
        orders = []

        # Convert the 'date' column from 'underlying_prices' to datetime, if necessary
        underlying_prices["date"] = pd.to_datetime(underlying_prices["date"])

        # Loop through each unique day and hour
        for date, hour in underlying_prices[["date", "hour"]].drop_duplicates().values:

            # Filter options for the current date and hour
            available_options = self.options[
                (self.options["day"] == date.date()) & 
                (self.options["hour"] == hour)
            ]

            if available_options.empty:
                print("No available options for the current date and hour")
                continue

            # Adjust the logic to ensure we're only looking at the 'C' or 'P' part of the symbol
            call_options = available_options[available_options["symbol"].str.contains(r'\d{6}C')]
            put_options = available_options[available_options["symbol"].str.contains(r'\d{6}P')]

            if call_options.empty or put_options.empty:
                print("No call or put options available")
                continue

            # Select the ATM (At-the-money) options (first one in each category)
            atm_call = call_options.iloc[0]
            atm_put = put_options.iloc[0]

            # Get the closing price for the current date and hour
            price = underlying_prices.loc[
                (underlying_prices["date"] == date) & 
                (underlying_prices["hour"] == hour), 
                "close"
            ].values[0]

            # Calculate delta for ATM options
            call_delta = self.calculate_delta(atm_call, price, date)
            put_delta = self.calculate_delta(atm_put, price, date)

            # Check for NaN values in call_delta or put_delta and skip the iteration if NaN is found
            if np.isnan(call_delta) or np.isnan(put_delta):
                print("Skipped - Delta calculation returned NaN values")
                continue

            # Initialize current delta if it's 0 (starting point)
            if self.current_delta == 0:
                self.current_delta = call_delta - put_delta

            # Vectorized logic for hedging
            if self.current_delta < 0:
                # Hedge by buying call options (positive delta to offset negative delta)
                order_size = min(int(atm_call["ask_sz_00"]), 5)
                orders.append({
                    "datetime": atm_call["ts_recv"],
                    "option_symbol": atm_call["symbol"],
                    "action": "B",  # Buy call
                    "order_size": order_size
                })
                self.current_delta += call_delta * order_size

            if self.current_delta > 0:
                # Hedge by buying put options (negative delta to offset positive delta)
                order_size = min(int(atm_put["ask_sz_00"]), 5)
                orders.append({
                    "datetime": atm_put["ts_recv"],
                    "option_symbol": atm_put["symbol"],
                    "action": "B",  # Buy put
                    "order_size": order_size
                })
                self.current_delta += put_delta * order_size

        return pd.DataFrame(orders)
