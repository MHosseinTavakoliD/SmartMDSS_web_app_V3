import requests
from datetime import datetime

# Example timestamp for a different date
start_timestamp = int(datetime(2023, 5, 21).timestamp())
end_timestamp = int(datetime(2023, 5, 22).timestamp())

url = (
    f"https://ag.us.clearapis.com/v1.0/historical/hourly"
    f"?app_id=f3372d18&app_key=fed026edfd8bb6eecb7f9e36de057d31"
    f"&start={start_timestamp}&end={end_timestamp}&location=42.910198,-82.47973&unitcode=si-std"
)

response = requests.get(url)

if response.status_code == 200:
    weather_data = response.json()
    print("Hourly Weather Data:")
    for location, hourly_data in weather_data.items():
        for time_period, data in hourly_data.items():
            # Extract the timestamp from the time_period string
            timestamp = int(time_period.split(':')[1])
            human_time = datetime.utcfromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
            print(f"Time Period: {human_time}")
            for parameter, values in data.items():
                print(f"  {parameter}: {values}")
else:
    print(f"Failed to retrieve data: {response.status_code}, {response.text}")
