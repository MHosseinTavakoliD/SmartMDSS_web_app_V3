from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib import messages

MONGO_DATABASE_NAME = 'SmartMDSS_data'
MONGO_HOST = 'localhost'
MONGO_PORT = 27017


# Michigan hoe page:
def Michigan_home(request):
    client = get_mongo_client()
    db = client['SmartMDSS_data']
    collection = db['User_input_point']

    # Fetch shareable points for all visitors
    shared_points = list(collection.find({
        'share_point': True,
        'state': 'Michigan'
    }))

    # Convert the MongoDB objects to a JSON-like string for sharable points
    shared_points_json = dumps(shared_points)

    # Add additional context for user points if the user is authenticated
    context = {
        'shared_points': shared_points_json,
    }
    # print ("context", context)
    if request.user.is_authenticated:
        # Fetch user-specific points if logged in
        user_points = list(collection.find({'user_id': request.user.id,
                                            }))
        user_points_json = dumps(user_points)
        context['user_points'] = user_points_json
    return render(request, 'web_smartmdss/Michigan_home.html', context)


def Mobile_Michigan_home(request):
    client = get_mongo_client()
    db = client['SmartMDSS_data']
    collection = db['User_input_point']

    # Fetch shareable points for all visitors
    shared_points = list(collection.find({
        'share_point': True,
        'state': 'Michigan'
    }))

    # Convert the MongoDB objects to a JSON-like string for sharable points
    shared_points_json = dumps(shared_points)

    # Add additional context for user points if the user is authenticated
    context = {
        'shared_points': shared_points_json,
    }
    # print ("context", context)
    if request.user.is_authenticated:
        # Fetch user-specific points if logged in
        user_points = list(collection.find({'user_id': request.user.id,
                                            }))
        user_points_json = dumps(user_points)
        context['user_points'] = user_points_json
    return render(request, 'web_smartmdss/Mobile_Michigan_home.html', context)
###########################################################
def home(request):
    return render(request, 'web_smartmdss/Michigan_home.html')

from django.conf import settings
from .forms import CustomUserCreationForm

######################## home view ################################################
from bson.json_util import dumps


def home(request):
    client = get_mongo_client()
    db = client['SmartMDSS_data']
    collection = db['User_input_point']

    # Fetch shareable points for all visitors
    shared_points = list(collection.find({'share_point': True}))

    # Convert the MongoDB objects to a JSON-like string for sharable points
    shared_points_json = dumps(shared_points)

    # Add additional context for user points if the user is authenticated
    context = {
        'shared_points': shared_points_json,
    }
    # print ("context", context)
    if request.user.is_authenticated:
        # Fetch user-specific points if logged in
        user_points = list(collection.find({'user_id': request.user.id}))
        user_points_json = dumps(user_points)
        context['user_points'] = user_points_json
    # print("context", context)
    return render(request, 'web_smartmdss/home.html', context)



####################### Register and Login #############################################
from Smartmdss.mongo_setup import get_mongo_client
import pymongo

def get_mongo_client():
    client = pymongo.MongoClient('mongodb://localhost:27017/')
    return client

def register_user(request):
    if request.method == 'POST':
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()  # This saves the user to the default database using Django's ORM
            client = get_mongo_client()
            # MongoDB handling
            client = get_mongo_client()
            db = client.SmartMDSS_data
            user_data_collection = db.User_data
            try:
                user_data_collection.insert_one({
                    'username': user.username,
                    'email': user.email,
                    'phone': form.cleaned_data.get('phone', ''),
                    'user_type': form.cleaned_data.get('user_type', ''),
                    'city': form.cleaned_data.get('city', ''),
                    'county': form.cleaned_data.get('county', ''),
                    'state': form.cleaned_data.get('state', ''),
                })

            except Exception as e:
                print("Error inserting into MongoDB:", e)

            print("Form is valid, saving user...")
            print("Data inserted into MongoDB:", {
                'username': user.username,
                'email': user.email,
                'phone': form.cleaned_data.get('phone', ''),
                'user_type': form.cleaned_data.get('user_type', ''),
                'city': form.cleaned_data.get('city', ''),
                'county': form.cleaned_data.get('county', ''),
                'state': form.cleaned_data.get('state', ''),
            })
            client.close()  # Ensure to close the connection

            messages.success(request, f"Account created successfully for {user.username}. You can now log in.")
            return redirect('login')
    else:
        form = CustomUserCreationForm()
    return render(request, 'web_smartmdss/register.html', {'form': form})


def login_user(request):
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                messages.info(request, f"You are now logged in as {username}.")
                next_url = request.POST.get('next') or 'home'  # Default redirection if 'next' isn't provided
                return redirect(next_url)
            else:
                messages.error(request, "Invalid username or password.")
        else:
            messages.error(request, "Invalid username or password.")

    form = AuthenticationForm()
    # Include 'next' parameter to context if present in the GET request
    next_url = request.GET.get('next', '')
    return render(request, "web_smartmdss/login.html", {"form": form, "next": next_url})

########################## insert points and delete ############################################################
from .forms import PointOfInterestForm

from django.contrib.auth.decorators import login_required

@login_required  # Ensure that the user must be logged in
def area_of_interest(request):
    client = get_mongo_client()
    db = client['SmartMDSS_data']
    collection = db['User_input_point']

    if request.method == 'POST':
        form = PointOfInterestForm(request.POST)
        if form.is_valid():
            point_data = {
                'user_id': request.user.id,
                'username': request.user.username,  # Save the username of the user
                'latitude': form.cleaned_data['latitude'],
                'longitude': form.cleaned_data['longitude'],
                'road_name': form.cleaned_data['road_name'],
                'road_type': form.cleaned_data['road_type'],
                'winter_maintenance': form.cleaned_data['winter_maintenance'],
                'operation_plowing': form.cleaned_data['operation_plowing'],
                'operation_salting': form.cleaned_data['operation_salting'],
                'salt_type': form.cleaned_data['salt_type'],
                'plowing_per_day': form.cleaned_data['plowing_per_day'],
                'salting_per_day': form.cleaned_data['salting_per_day'],
                'share_point': form.cleaned_data['share_point'],
            }
            collection.insert_one(point_data)
            return redirect('area_of_interest')
    else:
        points = list(collection.find({'user_id': request.user.id}))
        for point in points:
            point['id'] = str(point['_id'])
        form = PointOfInterestForm()  # Empty form for new entries

    return render(request, 'web_smartmdss/area_of_interest.html', {'form': form, 'points': points})



from django.http import HttpResponseNotAllowed
from pymongo import MongoClient, errors
from bson.objectid import ObjectId

def delete_point(request, point_id):
    if request.method == 'POST':
        client = MongoClient('localhost', 27017)
        db = client['SmartMDSS_data']
        collection = db['User_input_point']
        object_id = ObjectId(point_id)
        print("Using ObjectId for deletion:", object_id)  # Log the ObjectId
        try:
            result = collection.delete_one({'_id': object_id})
            print("Delete result:", result.deleted_count)
            if result.deleted_count == 1:
                print("Point successfully deleted.")
            else:
                print("No point was deleted, ID used:", object_id)
        except errors.InvalidId:
            print("Invalid MongoDB ID:", point_id)
            return HttpResponseNotAllowed("Invalid point ID")
        except Exception as e:
            print(f"An error occurred while deleting: {e}")
        finally:
            client.close()

        return redirect('area_of_interest')
    else:
        return HttpResponseNotAllowed("Only POST requests are allowed")

########################################### click on a point ######################################
from django.http import JsonResponse
from bs4 import BeautifulSoup
import requests
from bson.objectid import ObjectId

def get_weather_data(request, point_id):
    print  ("point_id", point_id)
    client = MongoClient('mongodb://localhost:27017/')
    db = client.SmartMDSS_data
    weather_data = db.Pulled_data_past48h.find_one({'point_id': ObjectId(point_id)})
    ######## ask for trafic info
    point_data = db.User_input_point.find_one({'_id': ObjectId(point_id)})
    Get_winter_data =db.Pulled_data_past48h.find_one({'Winter_mntnc_sugg': {'$exists': True, '$ne': []}})
    # image_data = db.Pulled_data_past48h.find_one({'Image_data': {'$exists': True, '$ne': []}})
    # print ("weather_data", weather_data)
    if point_data:
        lat, lon = str(point_data['latitude']), str(point_data['longitude'])
        # print (type(lat), type(lon))
        # Fetch traffic data
        #t5YquvDYfcgWQJiLPRXyYOOngoP84PDm
        resp_tf = requests.get(
            f"https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/xml?key=keyyyyyy&point={lat},{lon}&unit=mph")
        soup = BeautifulSoup(resp_tf.text, "lxml-xml")
        print ("resp_tf", resp_tf)
        # Parse traffic data
        current_speed = soup.find("currentSpeed").text if soup.find("currentSpeed") else "N/A"
        max_speed = soup.find("freeFlowSpeed").text if soup.find("freeFlowSpeed") else "N/A"
        road_closure = soup.find("roadClosure").text if soup.find("roadClosure") else "N/A"
        if road_closure == "false": road_closure= "No"
        if road_closure == "True": road_closure = "Check for Incidents"
        traffic_data = {
            'current_speed': current_speed,
            'max_speed': max_speed,
            'road_closure': road_closure
        }
        print ("traffic_data", traffic_data)
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
            'humidity': [],
            'wind_speed': [],
            'weather_description': [],
            'traffic': traffic_data
        }
        # Reverse historical data
        historical_data_reversed = list(reversed(weather_data.get('historical', [])))

        # Combine reversed historical and forecast data
        for entry in historical_data_reversed + weather_data.get('forecast', []):
            data['times'].append(entry['time'].strftime('%b %d, %I %p'))
            data['temperatures'].append(entry['temp'])
            data['surface_temperatures'].append(entry['surface_temp'])
            data['snow'].append(entry['snow'])
            data['humidity'].append(entry['humidity'])
            data['wind_speed'].append(entry['wind_speed'])
            data['weather_description'].append(entry['weather_description'])


        image_data = weather_data.get('Image_data', None)
        # print ("image_data", image_data)
        try:
            image_info = {
                'image_base64': image_data['image'],
                'classification': image_data['classification'],
                'confidence': image_data['confidence'],
                'last_update': image_data['last_update']
            }
        except:
            image_info = {
                'image_base64': '',
                'classification': 'Not available',
                'confidence': 'Not available',
                'last_update': 'Not available',
            }

        # print ("image_info", image_info)
        # Prepare the data dictionary to include image_info
        data['image_info'] = image_info

        if Get_winter_data:
            data.update(
                {
                    'maintenance_times': [],
                    'water': [],
                    'ice': [],
                    'salt': [],
                    'snow_withintervention': [],
                    'actions_': [],
                    'maintenance_times_Nointrvention': [],
                    'water_Nointrvention': [],
                    'ice_Nointrvention': [],
                    'salt_Nointrvention': [],
                    "snow_Nointrvention" : []

                }
            )
            Get_winter_data = db.Pulled_data_past48h.find_one({'point_id': ObjectId(point_id)})
            WinterMNT_data_reversed = list(Get_winter_data.get('Winter_mntnc_sugg', []))

            # WinterMNT_data_reversed = db.Pulled_data_past48h.find({'_id': {'$in': [ObjectId(entry_id) for entry_id in weather_data.get('Winter_mntnc_sugg', [])]}})

            for entry in WinterMNT_data_reversed:
                data['maintenance_times'].append(entry['time'].strftime('%b %d, %I %p'))
                data['water'].append(entry['initial_water'])
                data['ice'].append(entry['initial_ice'])
                data['salt'].append(entry['initial_Salt'])
                data['snow_withintervention'].append(entry['initial_Snow'])
                data['action_label'] = 'Plowing' if entry['action'] == 1 else ('Plowing and Salting' if entry['action'] == 2 else ('Salting' if entry['action'] == 3 else ''))
                data['actions_'].append(data['action_label'])

            # WinterNoIntvn_data_reversed = db.Pulled_data_past48h.find({'_id': {'$in': [ObjectId(entry_id) for entry_id in weather_data.get('Winter_cond_no_intrvn', [])]}})

            WinterNoIntvn_data_reversed = list(Get_winter_data.get('Winter_cond_no_intrvn', []))
            for entry in WinterNoIntvn_data_reversed:
                data['maintenance_times_Nointrvention'].append(entry['time'].strftime('%b %d, %I %p'))
                data['water_Nointrvention'].append(entry['initial_water'])
                data['ice_Nointrvention'].append(entry['initial_ice'])
                data['salt_Nointrvention'].append(entry['initial_Salt'])
                data['snow_Nointrvention'].append(entry['initial_Snow'])
        # print (data)

        # if image_data and 'Image_data' in image_data:
        #     image_info = {
        #         'image_base64': image_data['Image_data']['image'],
        #         'classification': image_data['Image_data']['classification'],
        #         'confidence': image_data['Image_data']['confidence']
        #     }
        # else:
        #     image_info = {
        #         'image_base64': '',
        #         'classification': 'Not available',
        #         'confidence': 'Not available'
        #     }
        #
        # # Prepare the data dictionary to include image_info
        # data['image_info'] = image_info

        return JsonResponse(data)
    else:
        return JsonResponse({'error': 'Data not found'}, status=404)


##########  Winter Demo Page ##########################################################
from django.http import JsonResponse
from pymongo import MongoClient
import datetime

def demo_data(request):
    client = MongoClient('mongodb://localhost:27017/')
    db = client['SmartMDSS_data']
    collection = db['Test_winter']

    forecast_data = collection.find_one()  # Adjust to find the right document

    if forecast_data:
        times = []
        air_temps = []
        surface_temps = []
        snow_data = []
        maintenance_times = []
        water = []
        ice = []
        salt = []
        snow = []
        actions = []
        maintenance_times_Nointrvention=[]
        water_Nointrvention =[]
        ice_Nointrvention=[]
        salt_Nointrvention=[]
        snow_Nointrvention=[]
        action_label_Nointrvention =[]
        actions_Nointrvention=[]

        # Process forecast data
        for entry in forecast_data.get('forecast', []):
            times.append(entry['MeasureTime'])
            air_temps.append(entry['Air TemperatureF'])
            surface_temps.append(entry['Surface TemperatureF'])
            snow_data.append(entry['snow'])

        # Process maintenance suggestions
        for entry in forecast_data.get('Winter_mntnc_sugg', []):
            maintenance_times.append(str(entry['time']))
            water.append(entry['initial_water'])
            ice.append(entry['initial_ice'])
            salt.append(entry['initial_Salt'])
            snow.append(entry['initial_Snow'])
            action_label = 'Plowing' if entry['action'] == 1 else ('Plowing and Salting' if entry['action'] == 2 else ('Salting' if entry['action'] == 3 else ''))
            actions.append(action_label)

        for entry in forecast_data.get('Winter_cond_no_intrvn', []):
            maintenance_times_Nointrvention.append(str(entry['time']))
            water_Nointrvention.append(entry['initial_water'])
            ice_Nointrvention.append(entry['initial_ice'])
            salt_Nointrvention.append(entry['initial_Salt'])
            snow_Nointrvention.append(entry['initial_Snow'])

        data = {
            'times': times,
            'air_temps': air_temps,
            "snow_data":snow_data,
            'surface_temps': surface_temps,
            'maintenance_times': maintenance_times,
            'water': water,
            'ice': ice,
            'salt': salt,
            'snow': snow,
            'actions': actions,
            'maintenance_times_Nointrvention':maintenance_times_Nointrvention,
            'water_Nointrvention': water_Nointrvention,
            'ice_Nointrvention': ice_Nointrvention,
            'salt_Nointrvention':salt_Nointrvention,
            'snow_Nointrvention':snow_Nointrvention,

        }
        # print (data)
        return JsonResponse(data)
    else:
        return JsonResponse({'error': 'Data not found'}, status=404)


############  Submit the feedbacks and store
from bson.errors import InvalidId
from django.http import HttpResponse

def submit_suggestion(request, point_id):
    if request.method == 'POST':
        print ("point_id", point_id)
        # point_id = request.POST.get('point_id')
        user_suggestion = request.POST.get('userSuggestion')
        date_ = request.POST.get('date')
        time_ = request.POST.get('time')
        maintenance_type = request.POST.get('type')
        user_suggestion = date_ + " " +time_ + " " + maintenance_type + " " + user_suggestion
        print ("user_suggestion", user_suggestion)
        client = get_mongo_client()
        db = client['SmartMDSS_data']
        feedback_logs = db['FeedBack_logs']
        pulled_data = db['Pulled_data_past48h']

        try:
            # Ensure point_id is a valid ObjectId
            oid = ObjectId(point_id)
        except InvalidId:
            return HttpResponse(status=404)

            # Fetch the relevant data from Pulled_data_past48h for the selected point
        point_data = pulled_data.find_one({'point_id': oid})
        #print ("point_data", point_data)

        if point_data:
            # Prepare the data to be logged
            log_entry = {
                'point_id': point_id,
                'user_suggestion': user_suggestion,
                'point_data': point_data
            }

            # Insert the combined data into FeedBack_logs
            feedback_logs.insert_one(log_entry)

            return HttpResponse(status=204)  # Ok
        else:
            return HttpResponse(status=404)  # Not found if no point data

    else:
        return HttpResponse('Only POST requests are allowed', status=405)
