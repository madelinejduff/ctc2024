import pandas as pd
from datetime import datetime, timedelta

class Strategy:
    
    def __init__(self, risk_free_rate=0.035, volatility=0.2) -> None:
        # Updated initial capital to match the competition's specified amount
        self.capital: float = 100_000_000
        self.portfolio: float = 0
        self.current_delta: float = 0  # Track current portfolio delta for hedging
        self.risk_free_rate = risk_free_rate  # Annual risk-free interest rate
        self.volatility = volatility  # Assumed constant volatility for the underlying
        self.transaction_cost = 0.50  # Per contract transaction cost
        self.slippage = 0.001  # Slippage is 0.1%

        # Load options data
        self.options: pd.DataFrame = pd.read_csv(r"data/cleaned_options_data.zip")
        
        # Convert 'ts_recv' to datetime and make it timezone-aware in UTC
        self.options["ts_clean"] = pd.to_datetime(self.options["ts_recv"], utc=True)
        self.options["day"] = self.options["ts_clean"].dt.strftime('%Y-%m-%d')
        self.options["hour"] = self.options["ts_clean"].dt.hour
        self.options["expiration"] = pd.to_datetime(
            self.options.symbol.apply(lambda x: x[6:12]), format="%y%m%d", utc=True
        )
        self.options["days_to_exp"] = (self.options["expiration"] - self.options["ts_clean"]).dt.days
        self.options["strike"] = self.options.symbol.apply(lambda x: int(x[-8:])/1000.0)
        self.options["is_call"] = self.options.symbol.apply(lambda x: x[-9] == "C")

        # Load underlying data
        self.underlying = pd.read_csv(r"data/underlying_data_hour.csv")
        self.underlying.columns = self.underlying.columns.str.lower()

        # Ensure the 'date' column is in UTC
        self.underlying["date"] = pd.to_datetime(self.underlying["date"], utc=True)

        # Floor the 'date' column to the hour and extract day and hour
        self.underlying["date"] = self.underlying["date"].dt.floor("H")
        self.underlying["day"] = self.underlying["date"].dt.strftime('%Y-%m-%d')
        self.underlying["hour"] = self.underlying["date"].dt.hour

        # Add 'opening_price' column for the opening price of the day
        self.underlying = self.underlying.sort_values('date')
        self.underlying['open'] = self.underlying['open'].astype(float)
        self.underlying['opening_price'] = self.underlying.groupby('day')['open'].transform('first')

        # Remove precomputed RSI to prevent look-ahead bias

    def calculate_rsi(self, price_series: pd.Series, period: int = 14) -> float:
        """
        Calculate the RSI value for the latest data point using past data up to that point.
        """
        # Calculate the price changes
        delta = price_series.diff()

        # Separate gains and losses
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        # Calculate the rolling averages of gains and losses
        avg_gain = gain.rolling(window=period, min_periods=period).mean().iloc[-1]
        avg_loss = loss.rolling(window=period, min_periods=period).mean().iloc[-1]

        # Handle division by zero
        if avg_loss == 0:
            return 100  # RSI is 100 if average loss is zero

        # Calculate the Relative Strength (RS)
        rs = avg_gain / avg_loss

        # Calculate RSI
        rsi = 100 - (100 / (1 + rs))

        # Return the RSI value
        return rsi if not pd.isna(rsi) else 50  # Neutral value if RSI is NaN

    def generate_orders(self) -> pd.DataFrame:
        """
        Generate orders based solely on the RSI indicator.
        Ensures compliance with competition rules and prevents look-ahead bias.
        """
        orders = []
        num_atm_options = 3  # Define how many closest ATM options to consider

        # Get the minimum date and skip trading on the first day
        min_date = self.underlying["date"].min()
        skip_days_threshold = min_date + pd.Timedelta(days=1)

        # Sort underlying data to ensure chronological order
        self.underlying = self.underlying.sort_values('date').reset_index(drop=True)

        # Initialize a variable to store past prices for RSI calculation
        past_prices = []

        for idx, row in self.underlying.iterrows():
            current_date = row["date"]

            # Append the current price to past_prices
            past_prices.append(row['open'])

            # Skip the first day based on the date
            if current_date < skip_days_threshold:
                continue

            current_price = row['open']
            current_day = row['day']
            current_hour = row['hour']
            opening_price = row['opening_price']

            # Compute RSI using past data up to the current time
            if len(past_prices) < 15:  # Need at least 15 data points to compute RSI with period=14
                continue
            price_series = pd.Series(past_prices)
            rsi_value = self.calculate_rsi(price_series, period=14)

            # Get options available for the current day and hour
            available_options = self.options[
                (self.options["day"] == current_day) &
                (self.options["hour"] == current_hour) &
                (self.options["days_to_exp"] > 1) &  # Do not trade on or after the day before expiration
                (self.options["days_to_exp"] < 7)    # Only select options with less than 7 days until expiration
            ]

            if available_options.empty:
                continue

            # Copy to avoid SettingWithCopyWarning
            available_options = available_options.copy()

            # Add a new column to store the difference between the strike and current price
            available_options['strike_diff'] = (available_options['strike'] - current_price).abs()

            # Sort by strike price difference and select the top 3 calls and top 3 puts
            closest_calls = available_options[available_options['is_call']].sort_values(by='strike_diff').head(num_atm_options)
            closest_puts = available_options[~available_options['is_call']].sort_values(by='strike_diff').head(num_atm_options)

            # Ensure that there are enough calls and puts to continue
            if closest_calls.empty and closest_puts.empty:
                continue

            # Generate orders based on RSI
            # Trading logic based on RSI
            if rsi_value < 30:  # Oversold condition, consider buying
                # Buy calls
                for _, call_option in closest_calls.iterrows():
                    # Ensure trades are not placed on or after the day before expiration
                    if call_option['expiration'] <= current_date + timedelta(days=1):
                        continue

                    # Check for sufficient ask size for buys
                    if pd.isna(call_option['ask_px_00']) or call_option['ask_sz_00'] < 1:
                        continue

                    # Calculate premiums (per contract)
                    call_premium_buy = float(call_option['ask_px_00']) * 100

                    # Calculate slippage (0.1% of order value)
                    call_slippage_buy = call_premium_buy * self.slippage

                    # Total cost for buying options
                    total_call_cost = call_premium_buy + call_slippage_buy + self.transaction_cost

                    # Check if capital is sufficient
                    if self.capital >= total_call_cost:
                        orders.append({
                            'datetime': call_option['ts_recv'],  # Output ts_recv directly
                            'option_symbol': call_option['symbol'],
                            'action': 'B',  # Buy call
                            'order_size': 1
                        })
                        self.capital -= total_call_cost  # Deduct total cost from capital

                # Buy puts
                for _, put_option in closest_puts.iterrows():
                    # Ensure trades are not placed on or after the day before expiration
                    if put_option['expiration'] <= current_date + timedelta(days=1):
                        continue

                    # Check for sufficient ask size for buys
                    if pd.isna(put_option['ask_px_00']) or put_option['ask_sz_00'] < 1:
                        continue

                    # Calculate premiums (per contract)
                    put_premium_buy = float(put_option['ask_px_00']) * 100

                    # Calculate slippage (0.1% of order value)
                    put_slippage_buy = put_premium_buy * self.slippage

                    # Total cost for buying options
                    total_put_cost = put_premium_buy + put_slippage_buy + self.transaction_cost

                    # Check if capital is sufficient
                    if self.capital >= total_put_cost:
                        orders.append({
                            'datetime': put_option['ts_recv'],  # Output ts_recv directly
                            'option_symbol': put_option['symbol'],
                            'action': 'B',  # Buy put
                            'order_size': 1
                        })
                        self.capital -= total_put_cost  # Deduct total cost from capital

            elif rsi_value > 70:  # Overbought condition, consider selling
                # Sell calls
                for _, call_option in closest_calls.iterrows():
                    # Ensure trades are not placed on or after the day before expiration
                    if call_option['expiration'] <= current_date + timedelta(days=1):
                        continue

                    # Check for sufficient bid size for sells
                    if pd.isna(call_option['bid_px_00']) or call_option['bid_sz_00'] < 1:
                        continue

                    # Calculate premiums (per contract)
                    call_premium_sell = float(call_option['bid_px_00']) * 100

                    # Calculate slippage (0.1% of order value)
                    call_slippage_sell = call_premium_sell * self.slippage

                    # Total proceeds from selling options (we receive money)
                    total_call_proceeds = call_premium_sell - call_slippage_sell - self.transaction_cost

                    # Margin requirements for selling options
                    call_margin_required_sell = call_premium_sell + 0.1 * opening_price * 100

                    # Check if capital is sufficient
                    if self.capital >= call_margin_required_sell:
                        orders.append({
                            'datetime': call_option['ts_recv'],
                            'option_symbol': call_option['symbol'],
                            'action': 'S',  # Sell call
                            'order_size': 1
                        })
                        self.capital += total_call_proceeds  # Add proceeds to capital
                        self.capital -= call_margin_required_sell  # Deduct margin requirement

                # Sell puts
                for _, put_option in closest_puts.iterrows():
                    # Ensure trades are not placed on or after the day before expiration
                    if put_option['expiration'] <= current_date + timedelta(days=1):
                        continue

                    # Check for sufficient bid size for sells
                    if pd.isna(put_option['bid_px_00']) or put_option['bid_sz_00'] < 1:
                        continue

                    # Calculate premiums (per contract)
                    put_premium_sell = float(put_option['bid_px_00']) * 100

                    # Calculate slippage (0.1% of order value)
                    put_slippage_sell = put_premium_sell * self.slippage

                    # Total proceeds from selling options (we receive money)
                    total_put_proceeds = put_premium_sell - put_slippage_sell - self.transaction_cost

                    # Margin requirements for selling options
                    put_margin_required_sell = put_premium_sell + 0.1 * put_option['strike'] * 100

                    # Check if capital is sufficient
                    if self.capital >= put_margin_required_sell:
                        orders.append({
                            'datetime': put_option['ts_recv'],
                            'option_symbol': put_option['symbol'],
                            'action': 'S',  # Sell put
                            'order_size': 1
                        })
                        self.capital += total_put_proceeds  # Add proceeds to capital
                        self.capital -= put_margin_required_sell  # Deduct margin requirement

        # Convert to DataFrame
        orders_df = pd.DataFrame(orders)

        # Limit total orders to 20 per day
        if not orders_df.empty:
            orders_df['day'] = orders_df['datetime'].apply(lambda x: x.split('T')[0])
            orders_df = orders_df.groupby('day').apply(lambda x: x.head(20)).reset_index(drop=True)
            orders_df = orders_df.drop(columns=['day'])

        print("Generated orders:")
        print(orders_df)

        return orders_df
