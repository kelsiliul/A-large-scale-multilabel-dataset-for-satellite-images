# A-large-scale-multilabel-dataset-for-satellite-images
A large-scale multilabel dataset for satellite images

This repo can be used to create a multulabel segmentation dataset for satellite images.
---

The code has three steps:

## Step-0: Preliminary

Download raw osm files using `download_osms.py`
 > python download_osms.py
This command will download all osm files in contiguous unites states. So 48 states+D.C. (no Hawaii and Alaska).

:warning: When testing do not download all states. Instead edit the list in variable `states` before downloading.

Use the following comand to extract osm files.
> bash unzip_osm.sh

---
## Step-1: Locate Multilabel Image
class_functions.py and efficient_v2.py are used for getting image centroids and segmentation. The input osm file needs to be in the same folder.

Run command: 
> python efficient_v2.py -i [input filename]

to get `[filename].csv` and `[filename].npz`

---
## Step-2: Download Images
`naip_downloader.py` uses centroids in the csv file from the previous method in `saved/[filename].csv`. 

Run command: 
> python naip_Downloader.py

to get images at scale, stored in directory `images`

## Step-3: Evaluate Annotation and Segmentation

`show.ipynb` is a notebook for examining annotations, and segmentation.
With images and segmentation in npz file,  show('ref_number') will give an overlayed image of segmentation on the original image and label out all segmentation.
