import random
import urllib.request
# from shapely.geometry import shape, Point
from skimage.exposure import rescale_intensity
import numpy as np
import ee
import argparse
import csv
import json
from multiprocessing.dummy import Pool, Lock
import os
from os.path import join, isdir, isfile
from os import mkdir, listdir
from collections import OrderedDict
import time
from datetime import datetime, timedelta
import warnings
warnings.simplefilter('ignore', UserWarning)
from tqdm import tqdm
from PIL import Image
from shapely.geometry import Polygon

class GeoSampler:

    def sample_point(self):
        raise NotImplementedError()


class NAIPSampler(GeoSampler):
    def __init__(self, rows):
        self.rows = rows
        self.fnames = [tmp[-1] for tmp in rows]

    def sample_point(self, idx):
        row = [self.fnames[idx], (float(self.rows[idx][3]), float(
            self.rows[idx][4])), '2008-01-01', '2024-4-1']
        return row

    def __iter__(self):
        return iter(self.fnames)

    def __len__(self):
        return len(self.fnames)

    @staticmethod
    def km2deg(kms, radius=6371):
        return kms / (2.0 * radius * np.pi / 360.0)


def get_collection():
    # collection = ee.ImageCollection('USDA/NAIP/DOQQ').select(['R','G'])
    collection = ee.ImageCollection("projects/planet-nicfi/assets/basemaps/asia").filterDate('2015-01-01', '2023-12-31')
    return collection

def cloudmask457(image):
	qa = image.select('QA60')
	mask = (qa.bitwiseAnd(1 << 10).eq(0)).And(qa.bitwiseAnd(1 << 11)).eq(0)
	return image.updateMask(mask).divide(10000)

# No Filter Needed as we are using the region function
def filter_collection(collection, coords, period=None, halfwidth=0.005):
    print("filtering collection")
    # Calculate the bounding box coordinates
    min_lon = coords[0] - halfwidth
    max_lon = coords[0] + halfwidth
    min_lat = coords[1] - halfwidth
    max_lat = coords[1] + halfwidth

    # Create a bounding box geometry
    bounding_box = ee.Geometry.Rectangle([min_lon, min_lat, max_lon, max_lat])

    # Start with the initial collection
    filtered = collection

    if period is not None:
        filtered = filtered.filterDate(*period)  # filter time

    # Apply the bounding box filter
    filtered = filtered.filterBounds(bounding_box)

    size_of = filtered.size().getInfo()

    if size_of == 0:
        print("filtered size is 0")
        raise ee.EEException(
            f'ImageCollection.filter: No suitable images found in ({coords[1]:.4f}, {coords[0]:.4f}) between {period[0]} and {period[1]}.')
    return filtered,size_of


def get_patch(collection, coords, bands=None, scale=None, save_path=None, fname=None):
    # print("entered get_patch")
    # if isfile(join(save_path, fname.split('/')[-1])):
    #     print("file exists")
    #     return None
    if bands is None:
        bands = RGB_BANDS
    collection = collection.sort('system:time_start', False)

    halfwidth = 0.0012
    # Define the bounding box region
    min_lon = coords[0] - halfwidth
    max_lon = coords[0] + halfwidth
    min_lat = coords[1] - halfwidth
    max_lat = coords[1] + halfwidth
    region = ee.Geometry.Rectangle([[min_lon, min_lat], [max_lon, max_lat]])
    # get sentinel region
  
    try:
        get_corresponding_sentinel(0, region, save_path, fname)
    except Exception as e:
        print("Sentinel Image unavailable", e)


   
    center = region.centroid().getInfo()['coordinates']
    width = region.bounds().getInfo()['coordinates'][0][2][0] - region.bounds().getInfo()['coordinates'][0][0][0]
    height = region.bounds().getInfo()['coordinates'][0][2][1] - region.bounds().getInfo()['coordinates'][0][0][1]
    new_width = width * 10
    new_height = height * 10
    
    # imgs_region = ee.Image(collection.getRegion(region,scale=20))

    sentinel_rectangle = Polygon([(center[0] - new_width/2, center[1] - new_height/2), (center[0] + new_width/2, center[1] - new_height/2), (center[0] + new_width/2, center[1] + new_height/2), (center[0] - new_width/2, center[1] + new_height/2)])
  
   
    left = sentinel_rectangle.bounds[0]
    mid = (sentinel_rectangle.bounds[0] + sentinel_rectangle.bounds[2])/2
    right = sentinel_rectangle.bounds[2]
    top = sentinel_rectangle.bounds[3]
    midtop = (sentinel_rectangle.bounds[1] + sentinel_rectangle.bounds[3])/2
    bottom = sentinel_rectangle.bounds[1]

    
    new_save_path = join(save_path,fname[:-4])
    if not isdir(new_save_path):
        mkdir(new_save_path)
    # print("new_save_path",new_save_path)

    # new_region = ee.Geometry.Rectangle([center[0] - new_width/2, center[1] - new_height/2, center[0] + new_width/2, center[1] + new_height/2])
    # print("creating rectangles")
    naip_rectangles= []
    num_rectangles = 4
    
    bottom_left = ee.Geometry.BBox(left, bottom, mid, midtop)
    naip_rectangles.append(bottom_left)
    top_left = ee.Geometry.BBox(left, midtop, mid, top)
    naip_rectangles.append(top_left)
    bottom_right = ee.Geometry.BBox(mid, bottom, right, midtop)
    naip_rectangles.append(bottom_right)
    top_right = ee.Geometry.BBox(mid, midtop, right, top)
    naip_rectangles.append(top_right)
    # print("naip_rectangles",naip_rectangles)

    # collection = collection.toList(collection.size())
    # print("getting naip images")
    # get naip images for each rectangle
    for i in range(len(naip_rectangles)):
        
        planet_col = collection.select('R', 'G', 'B').first()

        try:
            # get url for img
            try:
                url = planet_col.getThumbURL({'min': 64, 'max': 5454, 'dimensions': 512,'gamma':1.8, 'region': naip_rectangles[i], 'format':'jpg','crs':'EPSG:4326'})
                # print(url)
            except Exception as e:
                print("No RGB or NRG bands available: ",e)
                print("Skipping Image")
                continue

            # download img
            # new save path = old save path/fname/rectangle number
            # check if the image already exists
            if isfile(join(new_save_path,str(i)+'.jpg')):
                # check if image is downloaded properly i.e. not more than 1/4 of the image is black
                img = np.array(Image.open(join(new_save_path,str(i)+'.jpg')))
                if np.mean(img==0) < 0.25:
                    print("Image already exists")
                    continue
            urllib.request.urlretrieve(url, join(new_save_path,str(i)+'.jpg')) # change i to img_id if you want to save the image with the image id as the name
            # print("downloaded at ",join(save_path, fname))
            # check if image is downloaded properly i.e. not more than 1/4 of the image is black
            img = np.array(Image.open(join(new_save_path,str(i)+'.jpg')))
            if np.mean(img==0) > 0.25:
                print("redownloading image")
                urllib.request.urlretrieve(url, join(new_save_path,str(i)+'.jpg'))
                img = np.array(Image.open(join(new_save_path,str(i)+'.jpg')))
                if np.mean(img==0) > 0.25:
                    print("Image is still black, skipping image")
        except Exception as e:
            print("Image unavailable", e)

   
    return None

def get_corresponding_sentinel(img_id, region, save_path, fname):
    print("getting sentinel image")
    #  modify save_path to save sentinel image
    # get second folder name
    save_path = save_path+'_sentinel'
    if not isdir(save_path):
        mkdir(save_path)
    # check if sentinel image already exists
    if isfile(join(save_path, fname)):
        # check if image is downloaded properly i.e. not more than 1/4 of the image is black
        img = np.array(Image.open(join(save_path, fname)))
        if np.mean(img==0) < 0.25:
            print("sentinel file exists")
            # return 
    collection = ee.ImageCollection('COPERNICUS/S2').filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 2)).filterDate('2023-01-01', '2023-12-31')
    collection = collection.map(cloudmask457)

    image = collection.median().select(['B4','B3','B2'])
    center = region.centroid().getInfo()['coordinates']
    width = region.bounds().getInfo()['coordinates'][0][2][0] - region.bounds().getInfo()['coordinates'][0][0][0]
    height = region.bounds().getInfo()['coordinates'][0][2][1] - region.bounds().getInfo()['coordinates'][0][0][1]
    new_width = width * 10
    new_height = height * 10

    new_region = ee.Geometry.Rectangle([center[0] - new_width/2, center[1] - new_height/2, center[0] + new_width/2, center[1] + new_height/2])
    
    
    # get url for img

    # url = image.getThumbURL({'name': fname, 'format': 'jpg','crs': 'EPSG:4326', 'region': new_region, 'min': 0, 'max': 0.3, 'gamma': 1.0, 'scale': 10})
    # print(url)
    # urllib.request.urlretrieve(url, join(save_path, fname))
    

    try:
        # print("getting url")
        url = image.getThumbURL({'name': fname, 'format': 'jpg','crs': 'EPSG:4326', 'region': new_region, 'min': 0, 'max': 0.3, 'gamma': 1.0, 'scale': 10})
        # print(url)
        # download img
        urllib.request.urlretrieve(url, join(save_path, fname))

        # check if image is downloaded properly i.e. not more than 1/4 of the image is black
        img = np.array(Image.open(join(save_path, fname)))
        if np.mean(img==0) > 0.25:
            print("redownloading image")
            urllib.request.urlretrieve(url, join(save_path, fname))
            img = np.array(Image.open(join(save_path, fname)))
            if np.mean(img==0) > 0.25:
                print("Image is still black, skipping image")
                return None
    except Exception as e:
        print("Sentinel Image unavailable", e)
        return None
    
def date2str(date):
    return date.strftime('%Y-%m-%d')


def get_period(date, days=10):
    date1 = date[0] - timedelta(days=days)
    date2 = date[1] + timedelta(days=days)
    return date1, date2


def get_patches(collection, coords, startdate, enddate, debug=False, halfwidth=0.005, **kwargs):
    # print("starting to get patches")
    period = (startdate, enddate)
    try:
        patches = get_patch(collection, coords, **kwargs)
    except Exception as e:
        if debug:
            print(e)
        # raise
        return None
    return patches


class Counter:
    def __init__(self, start=0):
        self.value = start
        self.lock = Lock()

    def update(self, delta=1):
        with self.lock:
            self.value += delta
            return self.value
        




if __name__ == '__main__':
    b4 = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--which', type=str, default="NAIP",
                        choices=['NAIP', 'Sentinel-2', 'Sentinel-2-Temporal'])
    parser.add_argument('--preview', action='store_true')
    parser.add_argument('--num_workers', type=int, default=32)
    parser.add_argument('--cloud_pct', type=int, default=10)
    parser.add_argument('--log_freq', type=int, default=100)
    parser.add_argument('--indices_file', type=str, default=None)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    ee.Initialize()
    collection = get_collection()

    scale = {'B1': 60, 'B2': 10, 'B3': 10, 'B4': 10, 'B5': 20, 'B6': 20,
             'B7': 20, 'B8': 10, 'B8A': 20, 'B9': 60, 'B11': 20, 'B12': 20}
    RGB_BANDS = ['B4', 'B3', 'B2']
    if args.which == "NAIP":
        Sampler = NAIPSampler
        save_path = 'images'
        scale = {'R': 1, 'G': 1, 'B': 1}
        RGB_BANDS = ['R', 'G', 'B']
        halfwidth = 0.0012

    if not isdir(save_path):
        mkdir(save_path)

    counter = Counter()
    print(time.time()-b4)


    idir='coords'
    files = sorted(listdir(idir))
    random.shuffle(files)
    cutoff = 0
    for file in files:
        if 'tennis' in file:
            cutoff = 100/(11000*11000)
        if 'swimming_pool' in file:
            cutoff = 100/(11000*11000)
        if 'roundabout' in file:
            cutoff = 400/(11000*11000)
        if 'runway' in file:
            cutoff = 100/(11000*11000)
        if not isdir(join(save_path, file.split('.')[0])):
            mkdir(join(save_path, file.split('.')[0]))
        rows = []
        with open(join(idir, file),encoding='cp1252') as ifd:
            try:
                reader = csv.reader(ifd, delimiter=',')
                for i, row in enumerate(reader):
                    if row!=[]:
                        halfwidth=float(row[0])**(.5)
                        if float(row[0]) > cutoff:
                            rows.append([None, None, None, float(row[2]), float(row[1]), '_'.join(
                                [str(i).zfill(5)]+[str(np.round(float(tmp), 6)) for tmp in row[1:3]]+[str(int(np.round(float(tmp), 6))) for tmp in row[3:]])+'.jpg'])
            except Exception as e:
                print("Funky Characters, No Problem!, going to next picture: ",e)
                continue

        # rows = rows[50:51]
        # print("rows",rows)
        sampler = Sampler(rows)
        print("Number of rows",len(sampler))

        def worker(idx):
            pts = sampler.sample_point(idx)
            patches = get_patches(collection, pts[1], pts[2], pts[3], bands=RGB_BANDS, scale=scale, debug=args.debug, save_path=join(save_path, file.split('.')[0]), fname=pts[0], halfwidth=halfwidth)
            return

        print(file, len(sampler))
        indices = range(len(sampler))
        print("reached here")

        if args.num_workers == 0:
            
            for i in tqdm(range(len(sampler))):
                print(i)
                worker(i)
            print("completed")
        else:
            with Pool(args.num_workers) as p:
                p.map(worker, tqdm(range(len(sampler))))
            print("completed")