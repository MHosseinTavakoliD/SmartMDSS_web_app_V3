import random
import urllib.request
import re
import os
import time
from datetime import datetime
from pymongo import MongoClient
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
#from tensorflow import keras
import keras
import numpy as np
import tensorflow as tf
import base64
from keras.models import load_model
from bson.objectid import ObjectId


def determine_image_name(points_collection, lat, lon):
    # Normalize input coordinates to float
    try:
        lat_float = float(lat)
        lon_float = float(lon)
    except ValueError:
        print(f"Invalid coordinates: lat={lat}, lon={lon}")
        return f"new_image_point_{lat}_{lon}"

    # Define a small tolerance for coordinate comparison
    tolerance = 0#0.00001

    # Query the database for a point within the tolerance range
    point = points_collection.find_one({
        "$or": [
            {"latitude": {"$gte": lat_float - tolerance, "$lte": lat_float + tolerance},
             "longitude": {"$gte": lon_float - tolerance, "$lte": lon_float + tolerance}},
            {"latitude": {"$in": [str(lat_float), lat]},
             "longitude": {"$in": [str(lon_float), lon]}}
        ]
    })

    if point:
        print ("Found point _id: ", str(point['_id']))
        return str(point['_id'])
    else:
        print("Did not Found point _id: ", f"new_image_point_{lat}_{lon}")
        return f"new_image_point_{lat}_{lon}"
def run_smartmdss_datascraper_mdot_images():
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')
    #chromedriver_path = 'C:/Users/zmx5fy/SmartMDSS/web_smartmdss/BackEnd/Michigan/chromedrive/chromedriver123.exe'
    chromedrive_path = '/chromedriver-linux64/chromedriver'
    service = Service(chromedriver_path)
    driver = webdriver.Chrome(service=service, options=chrome_options)

    driver.get('https://mdotjboss.state.mi.us/MiDrive/cameras')
    myclient = MongoClient("mongodb://localhost:27017/")
    mydb = myclient["SmartMDSS_data"]
    points_collection = mydb["User_input_point"]
    classification_collection = mydb["Pulled_data_past48h"]

    model_directory = './CNN/Saved ML model/CNNmodel.h5'
    model = load_model(model_directory)

    now = datetime.now()
    update_file_name = now.strftime("%Y_%b_%d_%I_%M_%p")
    scraped_images_backup = f'./Scraped_imagesBackups/{update_file_name}/'
    os.makedirs(scraped_images_backup, exist_ok=True)
    for i in range(1, 30):
        print ("page.. ", i)
        time.sleep(4)
        wait = WebDriverWait(driver, 20)
        main = wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="cameraList"]/tbody')))

        # Retrieve links and store latitude and longitude
        links = driver.find_elements(By.CSS_SELECTOR, "#cameraList > tbody > tr > td > a")
        lat_lon_data = []
        print ("Getting links")
        for idx, link in enumerate(links):
            print ("idx, link",idx, link)
            try:
                href = link.get_attribute('href')
                print(href)
                lat = re.findall(r'lat=.*?&', href)
                print("lat 00", lat)
                lat = lat[0][4:-1]
                lon = re.findall(r'lon=.*?&', href)
                lon = lon[0][4:-1]
                print(lat, lon)



                name = determine_image_name(points_collection, lat, lon)
                print("Retrieve image")
                # Retrieve images
                images = main.find_elements(By.TAG_NAME, 'img')

                image = images[idx]
                print ("image", image)
                # lat, lon, name = lat_lon_data[idx]
                src = image.get_attribute('src')
                image_path = f'{scraped_images_backup}{name}.jpg'
                print ("Before")
                urllib.request.urlretrieve(src, image_path)
                print("After")
                # Perform image classification

                img = tf.keras.preprocessing.image.load_img(image_path, target_size=(150, 150))
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                img_array = np.expand_dims(img_array, 0)  # Adding a batch dimension

                # Correctly handling model prediction and output
                predictions = model.predict(img_array)
                score = tf.nn.softmax(predictions.squeeze())  # Assuming predictions is an array
                class_names = ['Heavy_snowy', 'Med_snowy', 'No_Image', 'No_Snowy_Road', 'Not_Clear', 'Wet_surface',
                               'slightly_snowy']
                class_prediction = class_names[np.argmax(score)]
                confidence = 100 * np.max(score)

                # Store classification result in database
                classification_collection.update_one(
                    {'point_id': ObjectId(name)},
                    {
                        '$set': {
                            'Image_data.last_update': now.strftime("%b %d %Y,%I%p"),
                            'Image_data.classification': class_prediction,
                            'Image_data.confidence': confidence,
                            'Image_data.image': base64.b64encode(open(image_path, "rb").read()).decode('utf-8')
                        }
                    },
                    upsert=True
                )
                print(f"Saved image as {image_path}, classified as: {class_prediction} with {confidence:.2f}% confidence")
                time.sleep(random.randint(1,4))
            except Exception as e:
                print(f"Error processing image at {href}: {e}")
        driver.find_element(By.XPATH, '//*[@id="cameraList_next"]/a').click()

    driver.quit()


run_smartmdss_datascraper_mdot_images()
