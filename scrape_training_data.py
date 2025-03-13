import os
import requests
from dotenv import load_dotenv


def main():
    marketstack_api_key = os.getenv("MARKETSTACK_API_KEY")

    # With query parameters
    params = {"access_key": marketstack_api_key, "symbols": "AAPL"}
    response = requests.get("http://api.marketstack.com/v2/eod/latest", params=params)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON response
        data = response.json()
        print(data)
    else:
        print(f"Request failed with status code: {response.status_code}")


if __name__ == "__main__":
    load_dotenv()
    main()
