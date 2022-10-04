# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 01:11:34 2022

@author: irfan
"""


import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np
from scipy.misc import imresize

#%matplotlib inline

import os
from os import listdir
from os.path import isfile, join
import shutil
import stat
import collections
from collections import defaultdict

from ipywidgets import interact, interactive, fixed
import ipywidgets as widgets

import h5py
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.applications.inception_v3 import preprocess_input
from keras.models import load_model



src_dir  = "D:/PhD_CourseWork/deep_learning/food101"
dest_dir = "D:/PhD_CourseWork/deep_learning/food101/working"
src_train = src_dir + '/working/train'
src_test = src_dir + '/working/test'

interest_class = ['apple_pie','baby_back_ribs','baklava','beef_carpaccio','beef_tartare','beet_salad','beignets','bibimbap','bread_pudding','breakfast_burrito','bruschetta','caesar_salad','cannoli','caprese_salad','carrot_cake','ceviche','cheesecake','cheese_plate','chicken_curry','chicken_quesadilla']


root_dir = "D:/PhD_CourseWork/deep_learning/food101/working/test21"
rows = 17
cols = 6
fig, ax = plt.subplots(rows, cols, frameon=False, figsize=(15, 25))
fig.suptitle('Random Image from Each Food Class', fontsize=20)
sorted_food_dirs = sorted(os.listdir(root_dir))
for i in range(rows):
    for j in range(cols):
        try:
            food_dir = sorted_food_dirs[i*cols + j]
        except:
            break
        all_files = os.listdir(os.path.join(root_dir, food_dir))
        rand_img = np.random.choice(all_files)
        img = plt.imread(os.path.join(root_dir, food_dir, rand_img))
        ax[i][j].imshow(img)
        ec = (0, .6, .1)
        fc = (0, .7, .2)
        ax[i][j].text(0, -20, food_dir, size=10, rotation=0,
                ha="left", va="top", 
                bbox=dict(boxstyle="round", ec=ec, fc=fc))
plt.setp(ax, xticks=[], yticks=[])
plt.tight_layout(rect=[0, 0.03, 1, 0.95])