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

        # Load underlying data
        self.underlying = pd.read_csv(r"data/underlying_data_hour.csv")
        self.underlying.columns = self.underlying.columns.str.lower()

        # Ensure the 'date' column is in UTC
        self.underlying["date"] = pd.to_datetime(self.underlying["date"], utc=True)

        # Floor the 'date' column to the hour and extract day and hour
        self.underlying["date"] = self.underlying["date"].dt.floor("H")
        self.underlying["day"] = self.underlying["date"].dt.strftime('%Y-%m-%d')
        self.underlying["hour"] = self.underlying["date"].dt.hour

        # Calculate RSI for underlying data
        self.underlying = self.underlying.sort_values('date')
        self.underlying['open'] = self.underlying['open'].astype(float)
        self.underlying['RSI'] = self.calculate_rsi(self.underlying, price_column='open', period=14)
        
    def calculate_rsi(self, data: pd.DataFrame, price_column: str, period: int = 14) -> pd.Series:
        """
        Calculate the Relative Strength Index (RSI) for a given data.
        Parameters:
        - data: DataFrame containing the data with a price column.
        - price_column: The name of the column containing price data.
        - period: The number of periods to use for the RSI calculation.
        Returns:
        - A pandas Series containing the RSI values.
        """
        # Ensure the timestamp is in datetime format and sort the data
        data = data.sort_values('date')
        
        # Calculate the price changes
        delta = data[price_column].diff()
        
        # Separate gains and losses
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        
        # Calculate the rolling averages of gains and losses
        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()
        
        # Calculate the Relative Strength (RS)
        rs = avg_gain / avg_loss
        
        # Calculate RSI
        rsi = 100 - (100 / (1 + rs))
        
        # Handle division by zero and fill initial periods
        rsi = rsi.fillna(50)  # Neutral value for RSI
        return rsi

    def generate_orders(self) -> pd.DataFrame:
        """
        Generate orders based on the RSI indicator with adjusted parameters to increase trade frequency and size.
        Buy or sell both calls and puts within a range of strike prices around the ATM.
        """

        orders = []
        num_atm_options = 5  # Increased from 3 to 5 to consider more options
        order_size = 10      # Increased order size from 1 to 10

        # Adjusted RSI thresholds
        rsi_buy_threshold = 40
        rsi_sell_threshold = 60

        # Remove the per-day order limit by commenting out the limiting code at the end

        # Loop through each row in the underlying data
        for idx, row in self.underlying.iterrows():
            current_date = row["date"]
            current_price = row['open']
            rsi_value = row['RSI']  # Get RSI value from underlying data
            current_day = row['day']
            current_hour = row['hour']

            # Get options available for the current day and hour
            available_options = self.options[
                (self.options["day"] == current_day) &
                (self.options["hour"] == current_hour)
                # Removed the days_to_exp constraint to include options with longer expirations
            ]

            if available_options.empty:
                continue

            # Copy to avoid SettingWithCopyWarning
            available_options = available_options.copy()

            # Add a new column to store the difference between the strike and current price
            available_options['strike_diff'] = (available_options['strike'] - current_price).abs()

            # Sort by strike price difference and select the top N calls and puts
            closest_calls = available_options[available_options['is_call']].sort_values(by='strike_diff').head(num_atm_options)
            closest_puts = available_options[~available_options['is_call']].sort_values(by='strike_diff').head(num_atm_options)

            # Ensure that there are enough calls and puts to continue
            if closest_calls.empty or closest_puts.empty:
                continue

            # Trading logic based on adjusted RSI thresholds
            if rsi_value < rsi_buy_threshold:  # Oversold condition, consider buying
                # Buy multiple calls and puts
                for _, option in pd.concat([closest_calls, closest_puts]).iterrows():
                    # Ensure trades are not placed on or after the day before expiration
                    if option['expiration'] <= current_date + timedelta(days=1):
                        continue

                    # Use 'ask_px_00' for buying
                    if pd.isna(option['ask_px_00']):
                        continue

                    option_price = float(option['ask_px_00'])
                    strike_price = option['strike']
                    margin_required = (option_price + 0.1 * strike_price) * 100 * order_size

                    # Skip if margin required is too high
                    if margin_required > self.capital:
                        continue

                    # Calculate slippage and transaction costs
                    slippage = margin_required * self.slippage
                    total_cost = margin_required + slippage + self.transaction_cost * order_size

                    if self.capital >= total_cost:
                        orders.append({
                            'datetime': option['ts_recv'],
                            'option_symbol': option['symbol'],
                            'action': 'B',  # Buy
                            'order_size': order_size
                        })
                        self.capital -= total_cost  # Deduct total cost from capital

            elif rsi_value > rsi_sell_threshold:  # Overbought condition, consider selling
                # Sell multiple calls and puts
                for _, option in pd.concat([closest_calls, closest_puts]).iterrows():
                    if option['expiration'] <= current_date + timedelta(days=1):
                        continue

                    # Use 'bid_px_00' for selling
                    if pd.isna(option['bid_px_00']):
                        continue

                    option_price = float(option['bid_px_00'])
                    strike_price = option['strike']
                    margin_required = (option_price + 0.1 * strike_price) * 100 * order_size

                    # Skip if margin required is too high
                    if margin_required > self.capital:
                        continue

                    # Calculate slippage and transaction costs
                    slippage = margin_required * self.slippage
                    total_credit = margin_required - slippage - self.transaction_cost * order_size

                    if self.capital >= margin_required:
                        orders.append({
                            'datetime': option['ts_recv'],
                            'option_symbol': option['symbol'],
                            'action': 'S',  # Sell
                            'order_size': order_size
                        })
                        self.capital += total_credit  # Add total credit to capital

        # Convert to DataFrame
        orders_df = pd.DataFrame(orders)

        # Remove the per-day order limit
        # Commented out the limiting code
        # if not orders_df.empty:
        #     orders_df['day'] = orders_df['datetime'].apply(lambda x: x.split('T')[0])
        #     orders_df = orders_df.groupby('day').apply(lambda x: x.head(20)).reset_index(drop=True)
        #     orders_df = orders_df.drop(columns=['day'])

        print("Generated orders:")
        print(orders_df)

        return orders_df
