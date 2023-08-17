# A-large-scale-multilabel-dataset-for-satellite-images
A large-scale multilabel dataset for satellite images

The code has three parts:

i). Locate Multilabel Image
---
class_functions.py and efficient_v2.py are used for getting image centroids and segmentation. The input osm file needs to be in the same folder.

Run command: python efficient_v2.py -i [input filename]

to get [filename].csv and [filename].npz

ii) Download Images
---
naip_downloader.py uses centroids in the csv file from the previous method in 'coords/[filename]'. 

Run command: python naip_Downloader.py

to get images at scale, stored in directory 'images'

iii) Evaluate Annotation and Segmentation
---
show.ipynb is a notebook for examining annotations, and segmentation.

With images and segmentation in npz file,  show('ref_number') will give an overlayed image of segmentation on the original image and label out all segmentation.
