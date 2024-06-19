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
import pickle
from shapely.geometry import Polygon
import ee
import numpy as np
import rasterio
import urllib3
from rasterio.transform import Affine
# from skimage.exposure import rescale_intensity
# from torchvision.datasets.utils import download_and_extract_archive
# import shapefile
from shapely.geometry import shape, Point
import urllib.request
import random

class GeoSampler:

    def sample_point(self):
        raise NotImplementedError()

class SentTempSampler(GeoSampler):
    def __init__(self, rows):
        self.rows = rows
        self.fnames = [tmp[0] for tmp in rows]

    def sample_point(self, idx):
        date = self.rows[idx][3].split(' ')[0]
        if int(date.split('-')[0])<=2018:
            # print(date)
            date = '2018'+date[4:]
            # print(date)
        row = [self.fnames[idx], (float(self.rows[idx][1]), float(self.rows[idx][2])), date]
        return row

    def __iter__(self):
        return iter(self.fnames)

    def __len__(self):
        return len(self.fnames)

    @staticmethod
    def km2deg(kms, radius=6371):
        return kms / (2.0 * radius * np.pi / 360.0)

def maskS2clouds(image):
    qa = image.select('QA60')
    # Bits 10 and 11 are clouds and cirrus, respectively.
    cloudBitMask = 1 << 10
    cirrusBitMask = 1 << 11
    # Both flags should be set to zero, indicating clear conditions.
    mask = qa.bitwiseAnd(cloudBitMask).eq(0)
    mask = mask.bitwiseAnd(cirrusBitMask).eq(0)
    return image.updateMask(mask)


def get_collection():
    collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    collection = collection.filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 5))
    collection = collection.filter(ee.Filter.lt('CLOUD_SHADOW_PERCENTAGE', 5))
    collection = collection.map(maskS2clouds)
    return collection


def filter_collection(collection, coords, period=None, halfwidth=0.005):
    filtered = collection
    if period is not None:
        filtered = filtered.filterDate(*period)  # filter time
    filtered = filtered.filterBounds(ee.Geometry.Point([coords[0]-halfwidth, coords[1]-halfwidth]))  # filter region
    filtered = filtered.filterBounds(ee.Geometry.Point([coords[0]-halfwidth, coords[1]+halfwidth]))  # filter region
    filtered = filtered.filterBounds(ee.Geometry.Point([coords[0]+halfwidth, coords[1]-halfwidth]))  # filter region
    filtered = filtered.filterBounds(ee.Geometry.Point([coords[0]+halfwidth, coords[1]+halfwidth]))  # filter region
    if filtered.size().getInfo() == 0:
        raise ee.EEException(
            f'ImageCollection.filter: No suitable images found in ({coords[1]:.4f}, {coords[0]:.4f}) between {period[0]} and {period[1]}.')
    return filtered


def get_properties(image):
    properties = {}
    for property in image.propertyNames().getInfo():
        properties[property] = image.get(property)
    return ee.Dictionary(properties).getInfo()


def get_patch(collection, coords, bands=None, scale=None, save_path=None,fname=None, date=None, planet_save_path=None, planet_collection=None):

    if isfile(join(save_path, fname.split('/')[-1])):
        return None
    if bands is None:
        bands = RGB_BANDS

    if args.which=="NAIP":
        collection = collection.sort('system:time_start', False)
    else:
        currtime = int(datetime.strptime(date, '%Y-%m-%d').timestamp()*1000)
        # print(currtime)
        def mapcloseness(image):
            return image.set('dateDist', ee.Number(image.get('system:time_start')).subtract(currtime).abs())
            # return image
        collection = collection.map(mapcloseness)
        collection = collection.sort('dateDist', True)

    # thanks to shitty ee api
    collection = collection.toList(collection.size())
    sentinel_halfwidth = 0.021
    sent_region = ee.Geometry.Rectangle([[coords[0]-sentinel_halfwidth, coords[1]-sentinel_halfwidth], [coords[0]+sentinel_halfwidth, coords[1]+sentinel_halfwidth]])
    region = ee.Geometry.Rectangle([[coords[0]-halfwidth, coords[1]-halfwidth], [coords[0]+halfwidth, coords[1]+halfwidth]])
    bbox_region = ee.Geometry.BBox(coords[0]-halfwidth, coords[1]-halfwidth, coords[0]+halfwidth, coords[1]+halfwidth)
    # # visualize region
    # import matplotlib.pyplot as plt
    sentinel_rectangle = Polygon([[coords[0]-halfwidth, coords[1]-halfwidth], [coords[0]+halfwidth, coords[1]-halfwidth], [coords[0]+halfwidth, coords[1]+halfwidth], [coords[0]-halfwidth, coords[1]+halfwidth]])
    # plt.plot(*sentinel_rectangle.exterior.xy, c='r')
    # print("length of side sentinel", sentinel_rectangle.bounds[2] - sentinel_rectangle.bounds[0])
   
    # sent_height = sentinel_rectangle.bounds[3] - sentinel_rectangle.bounds[1]
    # sent_width = sentinel_rectangle.bounds[2] - sentinel_rectangle.bounds[0]

    # planet_height = sent_height/4
    # planet_width = sent_width/4

    # planet_rectangle = Polygon([[coords[0]-planet_width, coords[1]-planet_height], [coords[0]+planet_width, coords[1]-planet_height], [coords[0]+planet_width, coords[1]+planet_height], [coords[0]-planet_width, coords[1]+planet_height]])
    # plt.plot(*planet_rectangle.exterior.xy, c='b')

    # 4x4 grid
    # divide sentinel rectangle into 4x4 grid
    left = sentinel_rectangle.bounds[0]
    mid = (sentinel_rectangle.bounds[0] + sentinel_rectangle.bounds[2])/2
    right = sentinel_rectangle.bounds[2]
    top = sentinel_rectangle.bounds[3]
    midtop = (sentinel_rectangle.bounds[1] + sentinel_rectangle.bounds[3])/2
    bottom = sentinel_rectangle.bounds[1]
    # # divide planet rectangle into 4x4 grid
    bottom_left = Polygon([[left, bottom], [mid, bottom], [mid, midtop], [left, midtop]])

    planet_regions = []
   
    bottom_left = ee.Geometry.BBox(left, bottom, mid, midtop)
    planet_regions.append(bottom_left)
    top_left = ee.Geometry.BBox(left, midtop, mid, top)
    planet_regions.append(top_left)
    bottom_right = ee.Geometry.BBox(mid, bottom, right, midtop)
    planet_regions.append(bottom_right)
    top_right = ee.Geometry.BBox(mid, midtop, right, top)
    planet_regions.append(top_right)
    

    outer_dir = save_path.split('/')[-1]
    planet_folder = fname.split('/')[-1].split('.')[0]
    planet_outer = join(planet_save_path, outer_dir)
    if not isdir(planet_outer):
        mkdir(planet_outer)
    planet_specific_save_path = join(planet_save_path, outer_dir, planet_folder)
    
    if not isdir(planet_specific_save_path):
        mkdir(planet_specific_save_path)
    planet_col = planet_collection.select('R', 'G', 'B').first()

    try:
        url = planet_col.getThumbURL({'min': 64, 'max': 5454, 'dimensions': 512,'gamma':1.8, 'region': bbox_region, 'format':'jpg','crs':'EPSG:4326'})
        # check if file exists
        if not isfile(join(save_path, fname.split('/')[-1])):
            urllib.request.urlretrieve(url, join(save_path, fname.split('/')[-1]))
        # print(url)
    except Exception as e:
        print(e)


    
   
    return None


def date2str(date):
    return date.strftime('%Y-%m-%d')


def get_period(date, days=10):
    date1 = date[0] - timedelta(days=days)
    date2 = date[1] + timedelta(days=days)
    return date1, date2


def get_patches_sent(collection, coords, date, debug=False, halfwidth=0.005, **kwargs):
    period = ('2018-01-01', '2023-12-31')
    # print(period)
    try:
        filtered_collection = filter_collection(collection, coords, period, halfwidth=halfwidth)
        planet_save_path = 'planetimageslg'
        if not isdir(planet_save_path):
            mkdir(planet_save_path)
        planet_collection =  ee.ImageCollection("projects/planet-nicfi/assets/basemaps/africa").filterDate('2023-01-01', '2023-12-31')
        patches = get_patch(filtered_collection, coords, date=date,planet_save_path=planet_save_path,planet_collection=planet_collection, **kwargs)
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
    parser.add_argument('--which', type=str, default="Sentinel-2-Temporal", choices=['NAIP', 'Sentinel-2', 'Sentinel-2-Temporal'])
    parser.add_argument('--preview', action='store_true')
    parser.add_argument('--num_workers', type=int, default=32)
    parser.add_argument('--cloud_pct', type=int, default=10)
    parser.add_argument('--log_freq', type=int, default=100)
    parser.add_argument('--indices_file', type=str, default=None)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    ee.Initialize()

    scale = {'B1': 60, 'B2': 10, 'B3': 10, 'B4': 10, 'B5': 20, 'B6': 20, 'B7': 20, 'B8': 10, 'B8A': 20, 'B9': 60, 'B11': 20, 'B12': 20}
    RGB_BANDS = ['B4', 'B3', 'B2']
    if args.which=="NAIP":
        Sampler = NAIPSampler
        save_path = 'naipimages'
        scale = {'R': 1, 'G': 1, 'B': 1}
        RGB_BANDS = ['R', 'G', 'B']
        halfwidth = 0.0021
    if args.which=="Sentinel-2-Temporal":
        collection = get_collection()
        Sampler = SentTempSampler
        save_path = 'images/planet_basemaps_africa'
        RGB_BANDS = ['B4', 'B3', 'B2']
        halfwidth = 0.0105

    if not isdir(save_path):
        mkdir(save_path)

    start_time = time.time()
    counter = Counter()
    print(time.time()-b4)

    with open('satellite_centers_planet.pkl', 'rb') as ifd:
        data = pickle.load(ifd)
        PoIs = data['PoIs']
        ImageIds = data['ImageIds']
        Dates = data['Dates']
    assert len(PoIs)==len(Dates)
    assert len(PoIs)==len(ImageIds)

    PoIs = [[str(ind).zfill(8)+'.jpg', poi[0], poi[1], Dates[ind]] for ind, poi in enumerate(PoIs)]

    binsize = 5000
    PoIss = [PoIs[i*binsize: min(i*binsize+binsize, len(PoIs))] for i in range((len(PoIs)-1)//binsize+1)]

    inds = list(range(len(PoIss)))
    random.shuffle(inds)

    for ind in inds:
        print(ind)
        PoIs = PoIss[ind]
        inter_dir = str(ind).zfill(4)
        if not isdir(join(save_path, inter_dir)):
            mkdir(join(save_path, inter_dir))
        rows = []
        print("Total Images:", len(PoIs))
        rows = [tmp for tmp in PoIs if not isfile(join(save_path, inter_dir, tmp[0]))]
        print("Images to be downloaded:", len(rows))
        random.shuffle(rows)

        sampler = Sampler(rows)
 
        def worker(idx):
            pts = sampler.sample_point(idx)
            # print(pts)
            # print(pts[0], pts[1], pts[2])
            patches = get_patches_sent(collection, pts[1], pts[2], bands=RGB_BANDS, scale=scale, debug=args.debug, save_path=join(save_path, inter_dir), fname=pts[0], halfwidth=halfwidth)
            return
        print(inter_dir, len(sampler))
        indices = range(len(sampler))


        if args.num_workers == 0:
            for i in indices:
                worker(i)
                break
        else:
            with Pool(args.num_workers) as p:
                p.map(worker, indices)
