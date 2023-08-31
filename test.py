import ee
import urllib.request

ee.Initialize()  
# Collection
col = ee.ImageCollection('USDA/NAIP/DOQQ').select(['R','G'])
# Region using coordinates
region = ee.Geometry.Rectangle([[-75.5870736080981, 39.572679390927405],[-75.5846736080981, 39.5750793909274]],bands=['R','G','B','N'])

# Get image using region
Region = ee.Image(col.getRegion(region,scale=20))
# complete path name
img = ee.Image('USDA/NAIP/DOQQ/'+Region.getInfo()[1820][0])

# get url for img
url = img.getThumbURL({'bands': ['R','G','B'], 'scale': 1, 'format': 'jpg',
                                'crs': 'EPSG:4326', 'region': region, 'min': 0, 'max': 255})
print(url)

# download img
urllib.request.urlretrieve(url, 'test.jpg')
