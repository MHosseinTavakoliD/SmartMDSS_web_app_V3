from apscheduler.schedulers.blocking import BlockingScheduler
import joblib
from Weather_API_Scraping import pull_weather_data
from XGBoost_surface_prediction import road_surface_temp_model
from pymongo import MongoClient
from Decision_making_DRL_for_homepage import execute_DRL_and_noIntervention
from Scrape_image_Michigan import run_smartmdss_datascraper_mdot_images
def modified_pull_weather_data():
    #  existing logic for pulling weather data
    pull_weather_data()
    # Call the model function right after pulling weather data
    road_surface_temp_model()
    # Call the DRL for Desion_Making
    execute_DRL_and_noIntervention()
    # Scrape images from Michigan website
    run_smartmdss_datascraper_mdot_images()



def start_scheduler():
    scheduler = BlockingScheduler()
    scheduler.add_job(modified_pull_weather_data, 'cron', minute=24)
    print(f"Scheduler started, job will run every hour at the 7th minute. ")
    scheduler.start()

start_scheduler()


# for i in range (1):
#     pull_weather_data()

###clear all the data in Pulled_data_past48h

# client = MongoClient('localhost', 27017)
# db = client.SmartMDSS_data
# points_collection = db.User_input_point
# weather_data_collection = db.Pulled_data_past48h
#
# result = weather_data_collection.delete_many({})
# print(f"Deleted {result.deleted_count} documents")