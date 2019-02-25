# this code simulates the time-of-flight data
# all time unit are picoseconds (1 picosec = 1e-12 sec)
import sys
sys.path.insert(0,'../pipe/')
import numpy as np
import os, json, glob
import imageio
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utils import *
from tof_class import *
import pdb
import pickle
import time
import scipy.misc
from scipy import sparse
import scipy.interpolate
from copy import deepcopy
from joblib import Parallel, delayed
import multiprocessing
from kinect_spec import *
import cv2
from numpy import linalg as LA

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
tf.logging.set_verbosity(tf.logging.INFO)

from kinect_init import *
from training_MOM import select_objects, data_augment_th

PI = 3.14159265358979323846

dtype = tf.float32

def gen_motion(array_dir, tof_cam):
    scenes = select_objects()
    x, y, _, Z_new = data_augment_th(scenes, array_dir, tof_cam)
    meas = x[:,:,0:9]
    msk = x[:,:,9:18]
    vy = y[:,:,0:9]
    vx = x[:,:,0:9]
    Z = Z_new[:,:,4]

    return meas, msk, vy, vx, Z

if __name__ == '__main__':
    # data
    array_dir = '../FLAT/trans_render/static/'
    data_dir = '../FLAT/kinect/'

    # initialize the camera model
    tof_cam = kinect_real_tf()

    # generate motion functions
    gen_motion(array_dir, tof_cam)