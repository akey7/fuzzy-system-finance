import os
import json
import requests
from dotenv import load_dotenv


def main():
    marketstack_api_key = os.getenv("MARKETSTACK_API_KEY")

    # With query parameters
    params = {
        "access_key": marketstack_api_key,
        "symbols": "AAPL",
        "date_from": "2025-01-02",
        "date_to": "2025-03-12",
        "limit": 100,
    }
    response = requests.get("http://api.marketstack.com/v2/eod", params=params)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON response
        data = response.json()
        output_filename = os.path.join("output", "latest_request.json")
        with open(output_filename, "w") as file:
            json.dump(data, file, indent=4, sort_keys=True)
        print(data)
    else:
        print(f"Request failed with status code: {response.status_code}")


if __name__ == "__main__":
    load_dotenv()
    main()
