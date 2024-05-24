import time

import requests
from pymongo import MongoClient
import datetime
from apscheduler.schedulers.blocking import BlockingScheduler
import pandas as pd
import numpy as np
import joblib

def road_surface_temp_model():
    client = MongoClient('mongodb://localhost:27017/')
    db = client.SmartMDSS_data
    weather_data_collection = db.Pulled_data_past48h
    user_input_collection = db.User_input_point
    documents_to_update = []

    for document in weather_data_collection.find():
        historical_data = document['historical']
        if len(historical_data) < 24:
            break

        forecast_data = document['forecast']
        # point_id = document['point_id']['$oid']
        user_input = user_input_collection.find_one({'point_id': document['_id']})
        if user_input and 'latitude' in user_input and 'longitude' in user_input:
            latitude = user_input['latitude']
            longitude = user_input['longitude']
        else:
            latitude = 49
            longitude = -85

        data_to_process = []
        for entry in historical_data[:24]:# Ensure using the first 24 hour
            data_to_process.append({
                "MeasureTime": entry["time"],
                "Rel. Humidity%": entry["humidity"],
                "Air TemperatureF": entry["temp"],
                "Surface TemperatureF": entry.get("surface_temp", None),
                "Wind Speed (act)mph": entry["wind_speed"],
                "Precipitation Intensityin/h": entry.get("rain", 0) + entry.get("snow", 0),
                "Station_name": str(document["point_id"]),
                "Latitude": latitude,
                "Longitude": longitude
            })
        for entry in forecast_data[:24]:
            data_to_process.append({
                "MeasureTime": entry["time"],
                "Rel. Humidity%": entry["humidity"],
                "Air TemperatureF": entry["temp"],
                "Surface TemperatureF": entry.get("surface_temp", None),
                "Wind Speed (act)mph": entry["wind_speed"],
                "Precipitation Intensityin/h": entry.get("rain", 0) + entry.get("snow", 0),
                "Station_name": str(document["point_id"]),
                "Latitude": latitude,
                "Longitude": longitude
            })


# Create DataFrame
        df = pd.DataFrame(data_to_process)
        df['MeasureTime'] = pd.to_datetime(df['MeasureTime'])

        # Save DataFrame to CSV
        df.to_csv('Time_frame_data_for_XGBoost_prediction.csv', index=False)
        #Read from the file
        new_df = pd.read_csv('DRL_models_soph/Time_frame_data_for_XGBoost_prediction.csv')
        new_df['MeasureTime'] = pd.to_datetime(new_df['MeasureTime'])

        # Define function to encode cyclical features
        def encode_cyclical_feature(df, col, max_vals):
            df[col + '_sin'] = np.sin(2 * np.pi * df[col] / max_vals)
            df[col + '_cos'] = np.cos(2 * np.pi * df[col] / max_vals)
            return df

        # Extract and encode cyclical time components
        new_df['hour'] = new_df['MeasureTime'].dt.hour
        new_df['day_of_week'] = new_df['MeasureTime'].dt.dayofweek
        new_df['day_of_month'] = new_df['MeasureTime'].dt.day
        new_df['month'] = new_df['MeasureTime'].dt.month
        new_df['year'] = new_df['MeasureTime'].dt.year
        new_df = encode_cyclical_feature(new_df, 'hour', 24)
        new_df = encode_cyclical_feature(new_df, 'day_of_week', 7)
        new_df = encode_cyclical_feature(new_df, 'month', 12)
        new_df = new_df.drop(['MeasureTime', 'Station_name'], axis=1)
        # Assuming the first 24 rows are the history data
        historical_data = new_df.iloc[:24]  # First 24 hours as history
        forecast_data = new_df.iloc[24:]  # Next 24 hours as forecast

        # Drop 'Surface TemperatureF' from forecast data
        forecast_data = forecast_data.drop('Surface TemperatureF', axis=1)

        # Concatenate historical and forecast data
        X_new = pd.concat([historical_data, forecast_data], axis=0).values.flatten()

        # Load the saved model
        model = joblib.load('xgboost_model.pkl')

        # Predict the next 24 surface temperatures
        predicted_surface_temps = model.predict(X_new.reshape(1, -1))

        # Output the predictions
        print("Predicted Surface Temperatures for the next 24 hours:", predicted_surface_temps)
        # Update documents in MongoDB
        # Convert predicted_surface_temps from numpy array to list
        predicted_surface_temps = predicted_surface_temps.tolist()
        print  ("predicted_surface_temps", predicted_surface_temps[0])
        forecast_updates = {}
        for idx, temp in enumerate(predicted_surface_temps[0]):
            forecast_updates[f"forecast.{idx}.surface_temp"] = temp

            # Update the document in the database with the modified forecast temperatures
            weather_data_collection.update_one(
                {'_id': document['_id']},
                {'$set': forecast_updates}
            )

        print(f"Updated forecast surface temperatures for document ID {document['_id']}")
    print(f"XGBoost done at.{datetime.datetime.now().time()}")



