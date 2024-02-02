# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import requests
from bs4 import BeautifulSoup
import json
import pprint
import csv


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

ourl = "https://www.bexrealty.com/New-York/New-York/?page_number="
data = ["price", "bedroom", "bathroom", "sqft", "pool", "neighbourhood", "coordinate"]
c=0
tooOld = False
csv_file_path = "housingdata2.csv"

for x in range(1,40): #145 25
    print(x)
    url = ourl + str(x)
    r = requests.get(url)
    soup = BeautifulSoup(r.content, 'html.parser')
    house_info = soup.find_all('script', type='application/ld+json')
    for i in range(0,len(house_info)):
        json_data = json.loads(house_info[i].string)
        mainEntity = json_data[0].get("mainEntity", None)
        if mainEntity is None: 
            continue
        offers = mainEntity[1].get("offers", None)
        if offers is None:
            continue
        indivUrl = offers.get("url", None)
        if indivUrl is None:
            continue
        
        r1 = requests.get(indivUrl)
        soup1 = BeautifulSoup(r1.content, 'html.parser')
        house_info1 = soup1.find_all('script', type='application/ld+json')
        json_data = json.loads(house_info1[0].string)
#         pprint.pprint(json_data)
        pool, garage, neighbourhood = None, None, None
        amenity = json_data[0].get("amenityFeature", None)
        if amenity is None: 
            continue        
        for thing in amenity:
            if thing.get("name") == "Pool":
                pool = thing.get("value")
            elif thing.get("name") == "Neighborhood":
                neighbourhood = thing.get("value")

        offers = json_data[0].get("offers", None)
        if offers is None: continue 
        price = offers.get("price", None)
        if price is None or price == 0: continue 
        accomo = json_data[0].get("accommodationFloorPlan", None)
        if accomo is None: continue 
        bedroom = accomo.get("numberOfBedrooms", None)
        totalbathroom = accomo.get("numberOfFullBathrooms", None)
        halfbathroom = accomo.get("numberOfPartialBathrooms", None)
        floorSize = accomo.get("floorSize", None)
        if bedroom is None or totalbathroom is None or floorSize is None or halfbathroom is None: continue
        if type(totalbathroom) == str:
            print(totalbathroom)
        bathroom = int(totalbathroom) + (int(halfbathroom) * 0.5)
        
        sqft = floorSize.get("value",None)
        if sqft is None or sqft == 0: continue 
        geo = json_data[0].get("geo", None)
        if geo is None: continue
        long = geo.get("longitude", None)
        lat = geo.get("latitude", None)

        if long is None or lat is None: continue
        coordinate = (long, lat)
        data.append([price,bedroom,bathroom,sqft,pool,neighbourhood,coordinate])
        
        c+=1
            
print(c)
with open(csv_file_path, mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    for row in data:
        csv_writer.writerow(row)
