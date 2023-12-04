# read through all images names in images/delaware-latest/

import os
import sys
import json
import time
import datetime
import numpy as np


files = os.listdir('images/vermont-latest/')
classes = []
labels = {}
count_classes = {}
# keep a list of first 5 files for each class
first5 = {}
# split by _

for file in files:
    # print(file)
    #  file name in format:  02420_38.602123_-75.252421_19_22_3_37.jpg
    # we want [19, 22, 3, 37]
    s = file[:-4]
    s = s.split('_')
    s = s[3:]
    classes.append(s)
    labels[file] = s
    for c in s:
        if c in count_classes:
            count_classes[c] += 1
        else:
            count_classes[c] = 1
    # first class is s[0]
    if s[0] in first5 and len(first5[s[0]]) < 5:
        first5[s[0]].append(file)
    elif s[0] not in first5:
        first5[s[0]] = [file]


# find out how many of each class there 
# get labels for each image

# sort count_classes by key
count_classes = sorted(count_classes.items(), key=lambda x: x[0])
print(count_classes)

print('first 5 files for each class')
# first5 = sorted(first5.items(), key=lambda x: x[0])
# print(first5)
# for c in first5:
#     print(c, first5[c])

# create a folder test_images for and have 5 images from each class

# if not os.path.exists('test_images'):
#     os.mkdir('test_images')

# #  copy 5 images from each class to test_images

# for c in first5:
#     for f in first5[c]:
#         os.system('cp images/delaware-latest/' + f + ' test_images/')
#     print(c, first5[c])


if not os.path.exists('test_images_sentinel'):
    os.mkdir('test_images_sentinel')


#  copy 5 images from each class to test_images

for c in first5:
    for f in first5[c]:
        os.system('cp images/delaware-latest/' + f + ' test_images_sentinel/')
    print(c, first5[c])
