import requests
from bs4 import BeautifulSoup
lat ='42.6'
lon= '-84.4'
resp_tf = requests.get(
            f"https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/xml?&key=7OCnm2rw9U511K8HfQObh1MWJD5XAGEi&point={lat},{lon}&unit=mph&openLr=true")
#https://api.tomtom.com/map/1/tile/basic/main/0/0/0.png?view=Unified&key=YOUR_API_KEY
# https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/xml?key={Your_API_Key}&point=52.41072,4.84239

soup = BeautifulSoup(resp_tf.text, "html.parser")
print ("resp_tf", resp_tf)