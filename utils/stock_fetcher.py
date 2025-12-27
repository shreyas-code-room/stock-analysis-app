# import requests
# import csv

# def fetch_stock_data(symbol):
#     api_key = "JOESOP4QIGQ1E2CG"
#     url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}.BSE&outputsize=full&apikey={api_key}'

#     response = requests.get(url)
#     data = response.json()

#     if "Time Series (Daily)" in data:
#         time_series = data["Time Series (Daily)"]
#         sorted_dates = sorted(time_series.keys())
#         stock_name = symbol.split('.')[0]
#         csv_file = f'{stock_name}.csv'

#         with open(csv_file, mode='w', newline='') as file:
#             writer = csv.writer(file)
#             writer.writerow(['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])

#             for date in sorted_dates:
#                 daily_data = time_series[date]
#                 row = [
#                     date,
#                     daily_data.get('1. open'),
#                     daily_data.get('2. high'),
#                     daily_data.get('3. low'),
#                     daily_data.get('4. close'),
#                     daily_data.get('5. volume')
#                 ]
#                 writer.writerow(row)

#         return csv_file
#     else:
#         raise ValueError("Error: Time Series (Daily) data not found in the response")
#new gpt code 

import requests
import pandas as pd
import logging
import time

def fetch_stock_data(symbol, exchange=None):
    api_key = "JOESOP4QIGQ1E2CG"

    # Handle exchange suffix properly
    if exchange and exchange in ["BSE", "NSE"]:
        full_symbol = f"{symbol}.{exchange}"
    else:
        full_symbol = symbol  # For US/global stocks, no suffix

    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={full_symbol}&outputsize=full&apikey={api_key}"
    response = requests.get(url)
    data = response.json()

    # Debug raw response
    logging.debug(f"API raw response for {full_symbol}: {data}")

    if "Time Series (Daily)" in data:
        time_series = data["Time Series (Daily)"]
        df = pd.DataFrame.from_dict(time_series, orient="index")
        df.index.name = "Date"
        df.rename(columns={
            "1. open": "Open",
            "2. high": "High",
            "3. low": "Low",
            "4. close": "Close",
            "5. volume": "Volume"
        }, inplace=True)
        df = df.astype({
            "Open": float,
            "High": float,
            "Low": float,
            "Close": float,
            "Volume": int
        })
        df.sort_index(inplace=True)

        filename = f"{full_symbol.replace('.', '_')}.csv"
        df.to_csv(filename)
        logging.info(f"Saved CSV as {filename}")
        return filename

    else:
        error_message = data.get("Error Message") or data.get("Note") or "Unexpected API response"
        logging.error(f"Error fetching {full_symbol}: {error_message}")
        return None

# Example usage
# fetch_stock_data("RELIANCE", "BSE")
# fetch_stock_data("TCS", "NSE")
# fetch_stock_data("AAPL")
