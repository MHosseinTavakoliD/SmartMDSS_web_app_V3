import pandas as pd
import pymongo
from pymongo import MongoClient
"""
 in the first part I wanted to insert the data from a csv file to mongodb
 I have a database in DRL with slightly differnet format. I wanted to translated into the 
 format that we use in the web app 
 Load the CSV file
"""
# file_path = 'test_winter.csv'
# df = pd.read_csv(file_path)
#
# # Classify precipitation as 'snow' or 'rain' and store the precipitation amount
# df['snow'] = df.apply(lambda x: x['Precipitation Intensityin/h'] if x['Air TemperatureF'] <= 33 and x['Precipitation Intensityin/h'] > 0 else 0, axis=1)
# df['rain'] = df.apply(lambda x: x['Precipitation Intensityin/h'] if x['Air TemperatureF'] > 33 and x['Precipitation Intensityin/h'] > 0 else 0, axis=1)
#
# # Drop the original 'Precipitation Intensityin/h' column as it's no longer needed
# df.drop('Precipitation Intensityin/h', axis=1, inplace=True)
#
# # Prepare data for MongoDB insertion
# data_to_insert = df.to_dict(orient='records')
#
# # Setup MongoDB connection
# client = MongoClient('mongodb://localhost:27017/')  # Adjust the connection URL as needed
# db = client['SmartMDSS_data']
# collection = db['Test_winter']
#
# # Create a document named 'forecast' and insert the data
# forecast_document = {'forecast': data_to_insert}
# collection.insert_one(forecast_document)
#
# print("Data has been successfully inserted into MongoDB.")
#################### test ##########################################
# The second part is for converting the format in the mongodb to feed into the DRL model
# the DRL wants every 3 hours data in different units
import pandas as pd
import pymongo
from pymongo import MongoClient

# Setup MongoDB connection
client = MongoClient('mongodb://localhost:27017/')  # Adjust the connection URL as needed
db = client['SmartMDSS_data']
collection = db['Test_winter']

# Fetch data from MongoDB
data = collection.find_one({'forecast': {'$exists': True}})  # Assuming one document format
df = pd.DataFrame(data['forecast'])

# Convert data types and perform unit conversions
df['MeasureTime'] = pd.to_datetime(df['MeasureTime'])
df['Air TemperatureC'] = (df['Air TemperatureF'] - 32) * 5.0/9.0  # F to C
df['Surface TemperatureC'] = (df['Surface TemperatureF'] - 32) * 5.0/9.0  # F to C
df['Wind Speed (act)m/s'] = df['Wind Speed (act)mph'] * 0.44704  # mph to m/s
df['Precipitation Intensitym/3h'] = df[['snow', 'rain']].sum(axis=1)  * 25.4 * 3  # inches/h to mm/3h

# Drop the columns that are no longer necessary
df.drop(['Air TemperatureF', 'Surface TemperatureF', 'Wind Speed (act)mph', 'snow', 'rain'], axis=1, inplace=True)

# Set index to MeasureTime
df.set_index('MeasureTime', inplace=True)

# Calculate total duration and determine resampling period
total_duration = (df.index.max() - df.index.min())
step_duration = total_duration / 9  # Dividing by 9 gives 10 intervals

# Resample data into 10 equal intervals
df_resampled = df.resample(step_duration).mean().reset_index()

# Add the window ID (assuming it is constant as 'Appleton_window_1')
df_resampled['Window_ID'] = 'Appleton_window_1'

# Reorder columns to match the desired output
df_resampled = df_resampled[['MeasureTime', 'Window_ID', 'Air TemperatureC', 'Surface TemperatureC', 'Wind Speed (act)m/s', 'Precipitation Intensitym/3h']]
pd.set_option('display.max_rows', None)  # or set to df_resampled.shape[0]
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.colheader_justify', 'center')
print(df_resampled.head(10))

