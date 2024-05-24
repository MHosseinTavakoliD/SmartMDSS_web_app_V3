import time
import requests
from pymongo import MongoClient
import datetime
from apscheduler.schedulers.blocking import BlockingScheduler
import pandas as pd
import numpy as np
import joblib

def pull_weather_data():
    client = MongoClient('mongodb://localhost:27017/')
    db = client.SmartMDSS_data
    weather_data_collection = db.Pulled_data_past48h

    points = db.User_input_point.find({})
    #19b3914f3dee28785eafb91d9acd2688   second e89481831e71d66e3af000e8db5883ce
    for point in points:
        try:
            time.sleep(0.5)
            print("point", point['_id'])
            response = requests.get(
                f"http://api.openweathermap.org/data/2.5/onecall?lat={point['latitude']}&lon={point['longitude']}"
                "&exclude=minutely&units=imperial&appid=keyyyy"
            )
            weather_data = response.json()
            #print("11111",weather_data)
            # response = requests.get(
            #         f"http://api.openweathermap.org/data/2.5/forecast?lat={point['latitude']}&lon={point['longitude']}"
            #         "&exclude=minutely&units=imperial&appid=keyyy"
            #     )
            # weather_data = response.json()
            # print ("222222",weather_data)

            current_weather = weather_data.get('current', {})
            hourly_forecast = weather_data.get('hourly', [])[1:49]  # Get the next 48 hours
            existing_data = weather_data_collection.find_one({'point_id': point['_id']})  # Ensure this line is before its usage.


            formatted_current = {
                'time': datetime.datetime.fromtimestamp(current_weather.get('dt')),
                'temp': current_weather.get('temp'),
                'surface_temp': int(current_weather.get('temp')) + 2,
                'feels_like': current_weather.get('feels_like'),
                'pressure': current_weather.get('pressure'),
                'humidity': current_weather.get('humidity'),
                'dew_point': current_weather.get('dew_point'),
                'uvi': current_weather.get('uvi'),
                'clouds': current_weather.get('clouds'),
                'visibility': current_weather.get('visibility', 10000),
                'wind_speed': current_weather.get('wind_speed'),
                'wind_deg': current_weather.get('wind_deg'),
                'weather_description': current_weather.get('weather', [{}])[0].get('description', 'clear'),
                'rain': current_weather.get('rain', {'1h': 0}).get('1h', 0),
                'snow': current_weather.get('snow', {'1h': 0}).get('1h', 0)
            }

            # Forecast handling
            formatted_forecast = []
            for f in hourly_forecast:
                formatted_forecast.append({
                    'time': datetime.datetime.fromtimestamp(f['dt']),
                    'temp': f.get('temp'),
                    'surface_temp': int(f.get('temp')) + 2,
                    'feels_like': f.get('feels_like'),
                    'pressure': f.get('pressure'),
                    'humidity': f.get('humidity'),
                    'dew_point': f.get('dew_point'),
                    'uvi': f.get('uvi'),
                    'clouds': f.get('clouds'),
                    'visibility': f.get('visibility', 10000),
                    'wind_speed': f.get('wind_speed'),
                    'wind_deg': f.get('wind_deg'),
                    'weather_description': f['weather'][0]['description'] if 'weather' in f else 'clear',
                    'rain': f.get('rain', {'1h': 0}).get('1h', 0),
                    'snow': f.get('snow', {'1h': 0}).get('1h', 0)
                })

            # Database update logic
            if existing_data:
                weather_data_collection.update_one(
                    {'point_id': point['_id']},
                    {'$push': {
                        'historical': {
                            '$each': [formatted_current],
                            '$position': 0,
                            '$slice': 48

                        }
                    },
                    '$set': {
                        'current_weather': formatted_current,
                        'forecast': formatted_forecast,
                        'timestamp': datetime.datetime.now()
                    }}
                )
            else:
                weather_data_collection.insert_one({
                    'point_id': point['_id'],
                    'timestamp': datetime.datetime.now(),
                    'current_weather': formatted_current,
                    'forecast': formatted_forecast,
                    'historical': [formatted_current]
                })
        except Exception as e:
                print(f"Scraping weather API error:  {e}")

    print(f"Weather data updated and historical data managed at {datetime.datetime.now().time()}")

pull_weather_data()

