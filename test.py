import ee
# ee.Authenticate()
ee.Initialize()
region = ee.Geometry.Rectangle([[-75.5908736080981, 39.5688793909274],[-75.5808736080981, 39.578879390927405]])
# Load a Landsat image.
img = ee.ImageCollection('USDA/NAIP/DOQQ').getRegion(region,scale=1)
print(img.getInfo().get('features')[0].get('id'))
# Print image object WITHOUT call to getInfo(); prints serialized request instructions.
# print(img)

# region = ee.Geometry.Rectangle([[-75.5908736080981, 39.5688793909274],[-75.5808736080981, 39.578879390927405]])


# print('Image collection from a string:', img.getRegion(region, scale=20))

# img1 = ee.Image('COPERNICUS/S2_SR/20170328T083601_20170328T084228_T35RNK')
# img2 = ee.Image('COPERNICUS/S2_SR/20170328T083601_20170328T084228_T35RNL')
# img3 = ee.Image('COPERNICUS/S2_SR/20170328T083601_20170328T084228_T35RNM')
# print('Image collection from a list of images:',
#       ee.ImageCollection([img1, img2, img3]))

# print('Image collection from a single image:',
#       ee.ImageCollection(img1).getInfo())

# Print image object WITH call to getInfo(); prints image metadata.
# print(img.getInfo())
