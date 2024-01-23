import xml.etree.ElementTree as etree
import codecs
import csv
import time
import os
from class_functions import filterfuncs, output_files
import numpy as np
import numpy.linalg as la

from shapely.geometry import Polygon
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import ConvexHull
from shapely import LineString
from scipy.spatial.qhull import QhullError

from shapely import intersects, union, intersection
from shapely import coverage_union
from shapely.geometry import Point
from shapely.validation import make_valid
from collections import defaultdict
import argparse

np.random.seed(42)
starttime = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input-file', help='The name of the input OSM File', default='d.osm')
args = parser.parse_args()


# Parse osm to coordinates of polygons under 40 classes
def getcoords(filterfuncs):
    polygons = {i: [] for i in range(len(filterfuncs))}
    polygons_coords = {i: [] for i in range(len(filterfuncs))}
    counter = 0
    ways=[]
    for event, elem in etree.iterparse(pathOSM, events=('start', 'end')):
        if event == 'start':
            pass
        else:
            counter+=1
            if elem.tag=='node':
                elem.clear()
            if elem.tag=='relation':
                elem.clear()
            if elem.tag=='way':
                tags = {}
                subnodes = []
                for child in elem:
                    if child.tag=='tag':
                        tags[child.get('k')] = child.get('v')
                    if child.tag=='nd':
                        subnodes.append(int(child.get('ref')))
                for indf, filterfunc in enumerate(filterfuncs):
                    if filterfunc(tags):
                        # print(filterfunc)
                        polygons[indf].append(subnodes)
                elem.clear()
            if counter%1000000==0:
                print(sum([len(polygons[key]) for key in polygons.keys()]), counter)
            pass
    for i in range(len(filterfuncs)):
        for polys in polygons[i]:
            polygons_coords[i].append([nodes[nodeid] for nodeid in polys])
    return polygons_coords


# Merge polygons that are neighbors
def close_inds(coordss):
    centroids = []
    for coords in coordss:
        if len(coords)<=2:
            centroid = np.mean(coords, axis=0)
            centroid = centroid[0], centroid[1]
        else:
            centroid = Polygon(coords).centroid
            centroid = centroid.x, centroid.y
        centroids.append(centroid)
    centroids = np.array(centroids)
    nbrs = NearestNeighbors(radius=0.00163080482).fit(centroids)
    closeinds = [-1 for i in range(len(centroids))]
    counter = 0
    for i in range(len(centroids)):
        if closeinds[i]==-1:
            _, inds = nbrs.radius_neighbors(centroids[i:i+1], return_distance=True)
            for ind in inds[0]:
                closeinds[ind]=counter
            counter+=1
    return closeinds
def merge_areas(coordss):
    inds = close_inds(coordss)
    for i in range(len(coordss)):
        if len(coordss[i])<=2:
            coordss[i] = coordss[i][:]+[coordss[i][-1]]
    hulls = []
    areas = []
    polygons=[]
    for i in np.unique(inds):
        coords = [coordss[j] for j in np.argwhere(inds==i)[:, 0]]
        coords = np.concatenate(coords, axis=0)
        try:
            hull = ConvexHull(coords).vertices
        except QhullError as E:
            continue
        hulls.append([coords[tmp] for tmp in hull])
    centroids = []
    for hull in hulls:
        pgon = Polygon(hull)
        centroids.append([pgon.centroid.x, pgon.centroid.y])
        polygons.append(pgon)
        areas.append(pgon.area)
    areas = np.array(areas)
    centroids = np.array(centroids)
    return areas, centroids, polygons

# Process polygons
def get_data(coordss):
    polygons = []
    areas = []
    centroids = []
    for i in range(len(coordss)):
        coords=coordss[i]
        if len(coords)<=2:
            pass
        else:
            pgon = Polygon(coords)
            areas.append(pgon.area)
            centroids.append([pgon.centroid.x, pgon.centroid.y])
            polygons.append(pgon)
    areas = np.array(areas)
    centroids = np.array(centroids)
    return areas,centroids,polygons

# For Line Curves, process them differently
def get_line(coordss):
    polygons=[]
    areas=[]
    centroids = [] 
    c=[]
    
    for i in range(len(coordss)):
        coords=coordss[i]
        if len(coords)<=2:
            pass
        else:
            coords = np.array(LineString(coords).xy).transpose() 
            pgon = LineString(coords).buffer(0.00012)
            areas.append(pgon.area)
            centroid = np.mean(coords, axis=0)
            centroid = [centroid[0], centroid[1] ]
            centroids.append(centroid)
            polygons.append(pgon)
    areas = np.array(areas)
    centroids = np.array(centroids)
    return areas,centroids,polygons


# Find farthest exterior point to centroid.
def get_radius(polygon, centroid):
    vertices = np.array([polygon.exterior.coords.xy]).T
    dist_c2v = la.norm(vertices-np.expand_dims(centroid, axis=1), axis=1)
    radius = np.max(dist_c2v)
    return radius


# Parser
pathOSM = args.input_file
nodes = {}
counter = 0
for event, elem in etree.iterparse(pathOSM, events=('start', 'end')):
    if event == 'start':
        pass
    else:
        counter+=1
        if elem.tag=='node':
            nodes[int(elem.get('id'))] = [float(elem.get('lat')), float(elem.get('lon'))]
        if counter%1000000==0:
            print(len(nodes), counter)
        elem.clear()
        pass

assert len(filterfuncs)==len(output_files)
print("time to get nodes = {}s".format(time.time()-starttime))

coords = getcoords(filterfuncs)
print("time to get polygons = {}s".format(time.time()-starttime))


lenlist=[]
for i in range(len(filterfuncs)):
    # print(str(len(coords[i]))+ "  is of class number "+str(i))
    lenlist.append(len(coords[i]))
print("Total polygons before merging:", sum(lenlist))


# Coordinate Processing, Merging
polygons = []
flattened = []               # contains the centroid and class name
radii = []                  # distance between centroids and the farthest vertex
area = []
for i in range(len(coords)):
    if len(coords[i]):
        # print(i)
        if i==19:
            areas, centroids, pgons=merge_areas(coords[i])
        elif i==36 or i==38 or i==37:
            areas, centroids, pgons=get_line(coords[i])
        else:
            areas, centroids, pgons=get_data(coords[i])
        if len(pgons)>0:
            inds = np.argsort(areas)[::-1]
            ofile=output_files[i]
            with open(ofile, 'w') as ofd:
                writer = csv.writer(ofd)
                for ind in inds:
                    writer.writerow([areas[ind]]+centroids[ind].tolist())
            polygons.extend(pgons)
            centroids= np.insert(centroids, centroids.shape[1], i, axis=1)
            # print(centroids)
            for poly, cent in zip(pgons, centroids):
                radii.append(get_radius(poly, cent[:2]))
            for sublist in centroids:
                flattened.append(sublist.tolist())
flattened = np.array(flattened)
print("Total polygons after merging: ", len(flattened))
print("time to merge polygons = {}s".format(time.time()-starttime))

# Choose a threshold for least polygons to be in the inner loop
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
max_size=[]
l_array=[]
n_array=[]
radius=np.sort(radii)
epsilon = 0.001
loop=np.arange(0.9,1,0.001)
for i in loop:
    if int(i)==1:
        break
    l_inds=np.argsort(radii)[int(len(radii)*i):]
    maxsearchradius=radius[int(len(radii)*i)]+epsilon
    neighbors = NearestNeighbors(radius=maxsearchradius)
    neighbors.fit(flattened[:, :2])
    numnbrs = [neighbors.radius_neighbors(flattened[tmp:tmp+1, :2], return_distance=False)[0].shape[0] for tmp in np.random.randint(0, len(flattened), 100)]
    n_array.append(np.mean(numnbrs))
    numnbrs = np.mean(numnbrs)+len(l_inds)
    l_array.append(len(l_inds))
    max_size.append(numnbrs) 
# plt.figure()
# plt.plot(loop, max_size,label="sum")
# plt.plot(loop, l_array,label="large polys")
# plt.plot(loop, n_array,label="mean of neighbors ")
# plt.title('Choose a threshold')
# plt.xlabel('Percentile')
# plt.ylabel('Number of search polygons')
# plt.grid()
# plt.legend()
# plt.savefig('threshold.pdf')
print("optimal threshold : "+ str(loop[np.argmin(max_size)]))
print("optimal maxradius: "+str(radius[int(len(radii)*loop[np.argmin(max_size)])]))
print("max maxradius : "+ str(radius[-1]))

p=loop[np.argmin(max_size)]
epsilon = 0.001
radius=np.sort(radii)
l_inds=np.argsort(radii)[int(len(radii)*p):]
maxsearchradius=radius[int(len(radii)*p)]
print("Maxsearchradius = ", maxsearchradius)

neighbors = NearestNeighbors(radius=maxsearchradius)
neighbors.fit(flattened[:, :2])

numnbrs = [neighbors.radius_neighbors(flattened[tmp:tmp+1, :2], return_distance=False)[0].shape[0] for tmp in np.random.randint(0, len(flattened), 100)]
numnbrs = np.mean(numnbrs)
print("average number of neigbors for the first polygon:", numnbrs)

# Choose sampling centroid that satisfy: Multilabel, Good polygon Coverage for NAIP image
multicoords2_original = []
multicoords2_sentinel = []
counter_original = 0
counter_sentinel = 0
savedpolygons_original = []
savedpolygons_sentinel = []
savedpolygons2_original = {}
savedpolygons2_sentinel = {}

starttime = time.time()

for i in range(len(flattened)):

    halfwidth=0.0012
    other_points_original = []
    other_points_sentinel = []

    cent=flattened[i].tolist()
    # print(cent)

    center_pol=polygons[i]
    thisclass=int(flattened[i][2])
    centroid=flattened[i][:2]

    # original rectangle
    rectangle=Polygon([[centroid[0]-halfwidth, centroid[1]-halfwidth],[centroid[0]+halfwidth, centroid[1]-halfwidth],[centroid[0]+halfwidth, centroid[1]+halfwidth],[centroid[0]-halfwidth, centroid[1]+halfwidth]])         
    
    # sentinel rectangle is 10x the size of original rectangle
    new_width=(rectangle.bounds[2]-rectangle.bounds[0])*10
    new_height=(rectangle.bounds[3]-rectangle.bounds[1])*10
    sentinel_rectangle = Polygon([[centroid[0]-new_width/2, centroid[1]-new_height/2],[centroid[0]+new_width/2, centroid[1]-new_height/2],[centroid[0]+new_width/2, centroid[1]+new_height/2],[centroid[0]-new_width/2, centroid[1]+new_height/2]])
    
    if center_pol.is_simple== False:
        print("not simple")
        continue

    center_pol_original =intersection(center_pol,rectangle)
    this_pol_original = [p for p in center_pol_original.geoms] if center_pol_original.geom_type == 'MultiPolygon' else [center_pol_original]
    row_original = {int(thisclass): this_pol_original}

    # Process the sentinel rectangle
    center_pol_sentinel = intersection(center_pol, sentinel_rectangle)
    this_pol_sentinel = [p for p in center_pol_sentinel.geoms] if center_pol_sentinel.geom_type == 'MultiPolygon' else [center_pol_sentinel]
    row_sentinel = {int(thisclass): this_pol_sentinel}


    inds = neighbors.radius_neighbors(flattened[i:i+1, :2], return_distance=False)[0]
    inds = np.concatenate((inds, l_inds))

    for j in inds:
        if j==i:
            continue
        c = flattened[j][2]
        pol=polygons[j]

        # original
        if intersects(rectangle, pol):
            if pol.is_simple== False:
                print("not simple")
                continue
            pol2=intersection(pol, rectangle)
            th_pol = [q for q in pol2.geoms] if pol2.geom_type == 'MultiPolygon' else [pol2]
            row_original[int(c)] = th_pol
            this_pol_original = union(pol2, this_pol_original)
            other_points_original.extend([int(c)])

        # sentinel
        if intersects(sentinel_rectangle, pol):
            if pol.is_simple == False:
                print("not simple")
                continue
            pol2_sentinel = intersection(pol, sentinel_rectangle)
            th_pol_sentinel = [q for q in pol2_sentinel.geoms] if pol2_sentinel.geom_type == 'MultiPolygon' else [pol2_sentinel]
            row_sentinel[int(c)] = th_pol_sentinel
            this_pol_sentinel = union(pol2_sentinel, this_pol_sentinel)
            other_points_sentinel.extend([int(c)])

    if other_points_original != []:
        cent.extend(other_points_original)
        dictionary = dict.fromkeys(cent)
        deduplicated_list = list(dictionary)
        this_pol_original = intersection(this_pol_original, rectangle)
        s_original = this_pol_original[0]
        for p in this_pol_original.tolist():
            s_original = union(s_original, p)
        area_original = s_original.area
        ratio_original = area_original / rectangle.area
        if ratio_original > 0.7:
            savedpolygons_original.append(s_original)
            savedpolygons2_original[counter_original] = row_original
            multicoords2_original.append(deduplicated_list)
            counter_original += 1
    
    # Process the sentinel rectangle
    if other_points_sentinel != []:
        cent.extend(other_points_sentinel)
        dictionary = dict.fromkeys(cent)
        deduplicated_list = list(dictionary)
        this_pol_sentinel = intersection(this_pol_sentinel, sentinel_rectangle)
        s_sentinel = this_pol_sentinel[0]
        for p in this_pol_sentinel.tolist():
            s_sentinel = union(s_sentinel, p)
        area_sentinel = s_sentinel.area
        ratio_sentinel = area_sentinel / sentinel_rectangle.area
        if ratio_original > 0.7:
            savedpolygons_sentinel.append(s_sentinel)
            savedpolygons2_sentinel[counter_sentinel] = row_sentinel
            multicoords2_sentinel.append(deduplicated_list)
            counter_sentinel += 1


    if i%500==0:
        print("Processed {} out of {} centre points: {}%".format(i, len(flattened), np.round(i*100/(len(flattened)), 2)))
print("time to get many-to-one image coordinates = {}s".format(time.time()-starttime))


print("Length of savedpolygons_original:", len(savedpolygons_original))
print("Length of multicoords2_original:", len(multicoords2_original))

print("Length of savedpolygons_sentinel:", len(savedpolygons_sentinel))
print("Length of multicoords2_sentinel:", len(multicoords2_sentinel))

# Save Segmentation for the original rectangle
keys_original = list(savedpolygons2_original.keys())
values_original = list(savedpolygons2_original.values())
npz_file_path_original = "{}np.npz".format(args.input_file.split('/')[-1].split('.')[0])
np.savez(npz_file_path_original, keys=keys_original, values=values_original)

# Save Segmentation for the sentinel rectangle
keys_sentinel = list(savedpolygons2_sentinel.keys())
values_sentinel = list(savedpolygons2_sentinel.values())
npz_file_path_sentinel = "{}_sentinelnp.npz".format(args.input_file.split('/')[-1].split('.')[0])
np.savez(npz_file_path_sentinel, keys=keys_sentinel, values=values_sentinel)


# Save Sampling Centroid for the original rectangle
with open('{}_originalnp.csv'.format(args.input_file.split('/')[-1].split('.')[0]), 'w') as ofd_original:
    writer_original = csv.writer(ofd_original)
    for ind in range(len(multicoords2_original)):
        if len(multicoords2_original[ind]) >= 4:
            row = [savedpolygons_original[ind].area * (111000 * 111000)] + multicoords2_original[ind]
            writer_original.writerow(row)

# Save Sampling Centroid for the sentinel rectangle
with open('{}_sentinelnp.csv'.format(args.input_file.split('/')[-1].split('.')[0]), 'w') as ofd_sentinel:
    writer_sentinel = csv.writer(ofd_sentinel)
    for ind in range(len(multicoords2_sentinel)):
        if len(multicoords2_sentinel[ind]) >= 4:
            row = [savedpolygons_sentinel[ind].area * (111000 * 111000)] + multicoords2_sentinel[ind]
            writer_sentinel.writerow(row)

# get labels for same multi-coords but with larger rectangles, so cents are the exact same, there will be more labels and overlap


# # Choose sampling centroid that satisfy: Multilabel, Good polygon Coverage for NAIP image
# multicoords2=[]
# counter=0
# savedpolygons=[]
# savedpolygons2={}
# starttime = time.time()
# for i in range(len(flattened)):
#     halfwidth=0.0012
#     sentinel_halfwidth=halfwidth*10
#     other_points=[]
#     cent=flattened[i].tolist()
#     center_pol=polygons[i]
#     thisclass=int(flattened[i][2])
#     centroid=flattened[i][:2]
#     rectangle=Polygon([[centroid[0]-halfwidth, centroid[1]-halfwidth],[centroid[0]+halfwidth, centroid[1]-halfwidth],[centroid[0]+halfwidth, centroid[1]+halfwidth],[centroid[0]-halfwidth, centroid[1]+halfwidth]])         
#     # sentinel rectangle is 10x the size of original rectangle
#     new_width=(rectangle.bounds[2]-rectangle.bounds[0])*10
#     new_height=(rectangle.bounds[3]-rectangle.bounds[1])*10
#     sent_rect = Polygon([[centroid[0]-new_width/2, centroid[1]-new_height/2],[centroid[0]+new_width/2, centroid[1]-new_height/2],[centroid[0]+new_width/2, centroid[1]+new_height/2],[centroid[0]-new_width/2, centroid[1]+new_height/2]])
#     if center_pol.is_simple== False:
#         print("not simple")
#         continue
#     center_pol=intersection(center_pol,sent_rect)
#     this_pol=[]
#     if center_pol.geom_type == 'MultiPolygon':
#         for p in center_pol.geoms:
#             this_pol.append(p)
#     else:
#         this_pol=[center_pol]
#     row={}
#     if int(thisclass) in row:
#         row[int(thisclass)].extend(this_pol)
#     else:
#         row[int(thisclass)]=this_pol
#     inds = neighbors.radius_neighbors(flattened[i:i+1, :2], return_distance=False)[0]
#     inds=np.concatenate((inds,l_inds))
#     # print(inds)
#     for j in inds:
#         if j==i:
#             continue
#         c = flattened[j][2]
#         pol=polygons[j]
#         if intersects(sent_rect, pol):
#             if pol.is_simple== False:
#                 print("not simple")
#                 continue
#             pol2=intersection(pol, sent_rect)
#             th_pol=[]
#             if pol2.geom_type == 'MultiPolygon':
#                 for q in pol2.geoms:
#                     th_pol.append(q)
#             else:
#                 th_pol=[pol2]
#             if c in row:
#                 row[int(c)].extend(th_pol)
#             else:
#                 row[int(c)]=th_pol
#             this_pol=union(pol2,this_pol)
#             other_points.extend([int(c)])
#             dictionary = dict.fromkeys(other_points)
#             deduplicated_list = list(dictionary)
#     if other_points!=[] and deduplicated_list!=[int(thisclass)]:
#         cent.extend(deduplicated_list)
#         dictionary = dict.fromkeys(cent)
#         deduplicated_list = list(dictionary)
#         this_pol=intersection(this_pol,rectangle)
#         s=this_pol[0]
#         for p in this_pol.tolist():
#             s=union(s,p)
#         area=s.area
#         ratio= area/rectangle.area
#         if ratio>0.7:
#             savedpolygons.append(s)
#             savedpolygons2[len(savedpolygons)-1]=row
#             multicoords2.append(deduplicated_list)
#     if i%500==0:
#         print("Processed {} out of {} centre points: {}%".format(i, len(flattened), np.round(i*100/(len(flattened)), 2)))
# print("time to get many-to-one image coordinates = {}s".format(time.time()-starttime))


# print("Length of savedpolygons:", len(savedpolygons))
# print("Length of multicoords2:", len(multicoords2))

# # Save Sampling Centroid
# with open('{}_sentinel.csv'.format(args.input_file.split('/')[-1].split('.')[0]), 'w') as ofd:
#     writer = csv.writer(ofd)
#     for ind in range(len(multicoords2)):
#         if len(multicoords2[ind])>=4:
#             row=[savedpolygons[ind].area*(111000*111000)]+multicoords2[ind]
#             writer.writerow(row)


print("total time = {}s".format(time.time()-starttime))