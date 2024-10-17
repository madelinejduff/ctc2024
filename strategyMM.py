import pandas as pd
from datetime import datetime, timedelta

class Strategy:
  
    def __init__(self, risk_free_rate=0.03, volatility=0.2) -> None:
        self.capital: float = 100_000_000
        self.portfolio: float = 0
        self.current_delta: float = 0  # Track current portfolio delta for hedging
        self.risk_free_rate = risk_free_rate  # Annual risk-free interest rate
        self.volatility = volatility  # Assumed constant volatility for the underlying
        self.transaction_cost = 0.50  # Per contract transaction cost
        self.slippage = 0.001  # Slippage is 0.1%

        # Load options data
        self.options: pd.DataFrame = pd.read_csv(r"data/cleaned_options_data.zip")
        
        # Convert to UTC timezone
        self.options["ts_clean"] = pd.to_datetime(self.options["ts_recv"], utc=True)
        self.options["day"] = self.options["ts_clean"].dt.strftime('%Y-%m-%d')
        self.options["hour"] = self.options["ts_clean"].dt.hour
        self.options["expiration"] = pd.to_datetime(
            self.options.symbol.apply(lambda x: x[6:12]), format="%y%m%d", utc=True
        )
        self.options["days_to_exp"] = (self.options["expiration"] - self.options["ts_clean"]).dt.days
        self.options["strike"] = self.options.symbol.apply(lambda x: int(x[-8:])/1000.0)
        self.options["is_call"] = self.options.symbol.apply(lambda x: x[-9] == "C")

        # Calculate RSI for options data
        self.options["RSI"] = self.calculate_rsi(self.options, period=14)

        # Load underlying data
        self.underlying = pd.read_csv(r"data/underlying_data_hour.csv")
        self.underlying.columns = self.underlying.columns.str.lower()

        # Ensure the 'date' column is in UTC
        self.underlying["date"] = pd.to_datetime(self.underlying["date"], utc=True)

        # Floor the 'date' column to the hour and extract day and hour
        self.underlying["date"] = self.underlying["date"].dt.floor("H")
        self.underlying["day"] = self.underlying["date"].dt.strftime('%Y-%m-%d')
        self.underlying["hour"] = self.underlying["date"].dt.hour

        # Calculate the 48-period moving average (equivalent to 12 hours if data is at 15-minute intervals)
        self.underlying = self.underlying.sort_values('date')
        self.underlying['open'] = self.underlying['open'].astype(float)
        self.underlying['moving_average'] = self.underlying['open'].rolling(window=48).mean()
        
    def calculate_rsi(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate the Relative Strength Index (RSI) for a given stock data.
        Parameters:
        - data: DataFrame containing the stock data with 'ask_px_00' as the price.
        - period: The number of periods to use for the RSI calculation.
        Returns:
        - A pandas Series containing the RSI values.
        """
        # Ensure the timestamp is in datetime format and sort the data
        data = data.sort_values('ts_clean')
        
        # Calculate the price changes
        delta = data['ask_px_00'].diff()
        
        # Separate gains and losses
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        # Calculate the Relative Strength (RS)
        rs = gain / loss
        rs = rs.fillna(0)  # Handle division by zero
        
        # Calculate RSI
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def generate_orders(self) -> pd.DataFrame:
        """
        Generate market-neutral orders based on the moving average and RSI.
        Buy or sell both calls and puts within a range of strike prices around the ATM.
        Limit to 20 orders per day.
        Only trade options with less than 7 days until expiration, and do not trade on or the day before expiration.
        """
        orders = []
        num_atm_options = 3  # Define how many closest ATM options to consider

        # Get the minimum date and skip trading in the first 1 day
        min_date = self.underlying["date"].min()
        skip_days_threshold = min_date + pd.Timedelta(days=1)

        for idx, row in self.underlying.iterrows():
            current_date = row["date"]

            # Skip the first 1 day based on the date
            if current_date < skip_days_threshold:
                continue

            current_price = row['open']
            ma_price = row['moving_average']
            current_day = row['day']
            current_hour = row['hour']

            # Ensure moving_average is not NaN
            if pd.isna(ma_price):
                continue

            # Get options available for the current day and hour
            available_options = self.options[
                (self.options["day"] == current_day) &
                (self.options["hour"] == current_hour) &
                (self.options["days_to_exp"] < 7)  # Only select options with less than 7 days until expiration
            ]

            if available_options.empty:
                continue

            # Filter based on RSI (only consider options where RSI is above 70 or below 30)
            available_options = available_options[(available_options['RSI'] > 70) | (available_options['RSI'] < 30)]

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
            if closest_calls.empty or closest_puts.empty:
                continue

            # Generate orders based on the comparison of current price, moving average, and RSI
            for _, call_option in closest_calls.iterrows():
                for _, put_option in closest_puts.iterrows():
                    # Ensure trades are not placed past the day before expiration
                    if call_option['expiration'] <= current_date + timedelta(days=1):
                        continue
                    if put_option['expiration'] <= current_date + timedelta(days=1):
                        continue

                    # Calculate margin for both calls and puts
                    if pd.isna(call_option['ask_px_00']) or pd.isna(put_option['ask_px_00']):
                        continue

                    call_margin_required = (float(call_option['ask_px_00']) + 0.1 * current_price) * 100
                    put_margin_required = (float(put_option['ask_px_00']) + 0.1 * put_option['strike']) * 100

                    # Skip if margin required is too high
                    if call_margin_required > self.capital or put_margin_required > self.capital:
                        continue

                    # Calculate slippage and transaction costs
                    call_slippage = call_margin_required * self.slippage
                    put_slippage = put_margin_required * self.slippage

                    # Add transaction costs (per contract)
                    total_call_cost = call_margin_required + call_slippage + self.transaction_cost
                    total_put_cost = put_margin_required + put_slippage + self.transaction_cost

                    # Handle buy/sell logic based on available capital, moving average, and RSI comparison
                    if current_price > ma_price and available_options['RSI'].mean() < 30:  # RSI indicates oversold
                        # Uptrend: Buy both the call and the put if there is enough margin
                        if self.capital >= total_call_cost:
                            orders.append({
                                'datetime': call_option['ts_recv'],  # Output ts_recv directly
                                'option_symbol': call_option['symbol'],
                                'action': 'B',  # Buy call
                                'order_size': 1
                            })
                            self.capital -= total_call_cost

                        if self.capital >= total_put_cost:
                            orders.append({
                                'datetime': put_option['ts_recv'],  # Output ts_recv directly
                                'option_symbol': put_option['symbol'],
                                'action': 'B',  # Buy put
                                'order_size': 1
                            })
                            self.capital -= total_put_cost

                    elif current_price < ma_price and available_options['RSI'].mean() > 70:  # RSI indicates overbought
                        # Downtrend: Sell both the call and the put if there is enough margin
                        if self.capital >= total_call_cost:
                            orders.append({
                                'datetime': call_option['ts_recv'],  # Output ts_recv directly
                                'option_symbol': call_option['symbol'],
                                'action': 'S',  # Sell call
                                'order_size': 1
                            })
                            self.capital += total_call_cost

                        if self.capital >= total_put_cost:
                            orders.append({
                                'datetime': put_option['ts_recv'],  # Output ts_recv directly
                                'option_symbol': put_option['symbol'],
                                'action': 'S',  # Sell put
                                'order_size': 1
                            })
                            self.capital += total_put_cost

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