import ee
import urllib.request
import numpy as np
from numpy import mean
# Download img from NAIP (this works fine)
ee.Initialize()  
# Collection
col = ee.ImageCollection('USDA/NAIP/DOQQ').select(['R','G'])
# Region using coordinates
region = ee.Geometry.Rectangle([[-75.6870736080981, 39.672679390927405],[-75.6846736080981, 39.6750793909274]],bands=['R','G','B','N'])

# Get image using region
Region = ee.Image(col.getRegion(region,scale=20))
print(Region.getInfo()[1820][0])
# complete path name
img = ee.Image('USDA/NAIP/DOQQ/'+Region.getInfo()[1820][0])

# get url for img
url = img.getThumbURL({'bands': ['R','G','B'], 'scale': 1, 'format': 'jpg',
                                'crs': 'EPSG:4326', 'region': region, 'min': 0, 'max': 255})
print(url)

# download img
urllib.request.urlretrieve(url, 'test.jpg')
#---------------------------------------------------------------------------------------------------------

# download same image from sentinel

def cloudmask457(image):
	qa = image.select('QA60')
	mask = (qa.bitwiseAnd(1 << 10).eq(0)).And(qa.bitwiseAnd(1 << 11)).eq(0)
	return image.updateMask(mask).divide(10000)


# # Sentinel-2 Image Collection
region2 = ee.Geometry.Rectangle([[-75.6870736080981, 39.672679390927405],[-75.6846736080981, 39.6750793909274]])
#  expand region to get larger image by 10x
# region2 = region2.buffer(1000)


collection = ee.ImageCollection('COPERNICUS/S2').filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 2))
collection = collection.map(cloudmask457)

image = collection.median().select(['B4','B3','B2'])
center = region2.centroid().getInfo()['coordinates']
width = region2.bounds().getInfo()['coordinates'][0][2][0] - region2.bounds().getInfo()['coordinates'][0][0][0]
height = region2.bounds().getInfo()['coordinates'][0][2][1] - region2.bounds().getInfo()['coordinates'][0][0][1]
new_width = width * 10
new_height = height * 10

new_region2 = ee.Geometry.Rectangle([center[0] - new_width/2, center[1] - new_height/2, center[0] + new_width/2, center[1] + new_height/2])
# get url for img
url = image.getThumbURL({'name': "text2.jpg", 'format': 'jpg','crs': 'EPSG:4326', 'region': new_region2, 'min': 0, 'max': 0.3, 'gamma': 1.0, 'scale': 10})

print(url)
# download img
urllib.request.urlretrieve(url, 'test2.jpg')
