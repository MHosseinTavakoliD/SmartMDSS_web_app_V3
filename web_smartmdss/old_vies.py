from django.http import JsonResponse
from bs4 import BeautifulSoup
import requests
def get_weather_data(request, point_id):
    print  ("point_id", point_id)
    client = MongoClient('mongodb://localhost:27017/')
    db = client.SmartMDSS_data
    weather_data = db.Pulled_data_past48h.find_one({'point_id': ObjectId(point_id)})
    ######## ask for trafic info
    point_data = db.User_input_point.find_one({'_id': ObjectId(point_id)})
    Winter_Mntce_data =db.Pulled_data_past48h.find_one({'Winter_mntnc_sugg': {'$exists': True, '$ne': []}})

    # print ("weather_data", weather_data)
    if point_data:
        lat, lon = str(point_data['latitude']), str(point_data['longitude'])
        # print (type(lat), type(lon))
        # Fetch traffic data
        resp_tf = requests.get(
            f"https://api.tomtom.com/traffic/services/4/flowSegmentData/relative0/10/xml?point={lat},{lon}&unit=mph&openLr=true&key=7OCnm2rw9U511K8HfQObh1MWJD5XAGEi")
        soup = BeautifulSoup(resp_tf.text, "html.parser")
        print ("resp_tf", resp_tf)
        # Parse traffic data
        current_speed = soup.find("currentspeed").text if soup.find("currentspeed") else "N/A"
        max_speed = soup.find("freeflowspeed").text if soup.find("freeflowspeed") else "N/A"
        road_closure = soup.find("roadclosure").text if soup.find("roadclosure") else "N/A"

        traffic_data = {
            'current_speed': current_speed,
            'max_speed': max_speed,
            'road_closure': road_closure
        }

        # Store traffic data in MongoDB under the 'traffic' field
        db.Pulled_data_past48h.update_one({'_id': weather_data['_id']}, {'$set': {'traffic': traffic_data}},
                                          upsert=True)
    if weather_data:
        # Extract and format data for chart
        data = {
            'times': [],
            'temperatures': [],
            'surface_temperatures': [],
            'snow': [],
            'traffic': traffic_data
        }
        # Reverse historical data
        historical_data_reversed = list(reversed(weather_data.get('historical', [])))

        # Combine reversed historical and forecast data
        for entry in historical_data_reversed + weather_data.get('forecast', []):
            data['times'].append(entry['time'].strftime('%Y-%m-%d %H:%M:%S'))
            data['temperatures'].append(entry['temp'])
            data['surface_temperatures'].append(entry['surface_temp'])
            data['snow'].append(entry['snow'])

        # # Combine historical, current, and forecast data
        # for entry in weather_data.get('historical', []) +  weather_data.get(
        #         'forecast', []):
        #     data['times'].append(entry['time'].strftime('%Y-%m-%d %H:%M:%S'))
        #     data['temperatures'].append(entry['temp'])
        #     data['surface_temperatures'].append(entry['surface_temp'])
        #     data['snow'].append(entry['snow'])
        # print("data", data)
        if weather_data:
            data.update(
                {}
            )


        return JsonResponse(data)
    else:
        return JsonResponse({'error': 'Data not found'}, status=404)
