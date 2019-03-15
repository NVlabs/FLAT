# license:  Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
#           Licensed under the CC BY-NC-SA 4.0 license
#           (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode). 

# this code simulates the time-of-flight data of phasor
# all time unit are picoseconds (1 picosec = 1e-12 sec)
import numpy as np
import os, json, glob
import imageio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utils import *
from tof_class import *
import pdb
import pickle
import time
import scipy.misc
from shutil import copyfile
from scipy import sparse
from copy import deepcopy
from joblib import Parallel, delayed
import multiprocessing
import scipy.sparse as sp

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
tf.logging.set_verbosity(tf.logging.INFO)

def gen_raw(scene_n, data_dir, tof_cam, func):
	print('Processing scene', scene_n)

	# check if the file already exists
	save_dir = data_dir+scene_n[-23:-7]+'/'
	## load the data
	if 'data' not in locals():
		print('load data')
		with open(scene_n,'rb') as f:
			data = pickle.load(f)

	# copy the variables
	program = deepcopy(data['program'])
	cam = deepcopy(data['cam'])
	cam_t = deepcopy(data['cam_t'])
	scene = deepcopy(data['scene'])
	depth_true = deepcopy(data['depth_true'])
	prop_idx = deepcopy(data['prop_idx'])
	prop_s = deepcopy(data['prop_s'])

	# generate true depth
	depth_true_s = scipy.misc.imresize(depth_true,(cam['dimy'],cam['dimx']),mode='F')

	# generate the raw measurement
	res = func(cam, prop_idx, prop_s, scene, depth_true)
	meas = res['meas']

	# reshaping the images
	os.mkdir(save_dir)
	meas = meas - meas.min()
	meas = meas / (meas.max() - meas.min())
	meas = meas * 255
	meas = meas.astype(np.uint8)
	for i in range(8):cv2.imwrite(save_dir+str(i)+'.png', meas[:,:,i])  

	return

def gen_dataset(setup):
	"""
	Create compact array from the dataset
	"""
	data_dir = '../FLAT/trans_render/static/'
	save_dir = '../FLAT/'+setup + '/'
	sub_dirs = [
		'full/',   # raw measurements with both noise and reflection
		'noise/',  # raw measurements with only noise
		'reflection/',  # raw measurements with only reflection
		'ideal/' # raw measurements with no noise nor reflection
	]

	if not os.path.exists(data_dir):
		os.mkdir(data_dir)

	if not os.path.exists(save_dir):
		os.mkdir(save_dir)

	for sub_dir in sub_dirs:
		if not os.path.exists(save_dir+sub_dir):
			os.mkdir(save_dir+sub_dir)

	tof_cam = eval(setup+'()')

	funcs = [\
		tof_cam.process_gain_noise, 
		tof_cam.process_gt_gain_noise,
		tof_cam.process_gain,
		tof_cam.process_gt_gain,
	]

	"""
	Create raw measurements from compact array
	"""
	# input the folder that contains the data
	scenes = glob.glob(data_dir+'*.pickle')

	for j in range(len(sub_dirs)):
		sub_dir = sub_dirs[j]
		# jump over those already finished 
		scenes_finished = glob.glob(save_dir+sub_dir+'*')
		scenes_finished = [scene[-16::] for scene in scenes_finished]

		for i in range(len(scenes)):
			if (scenes[i][-23:-7] not in scenes_finished):
				gen_raw(scenes[i], save_dir+sub_dir, tof_cam, funcs[j])

	"""
	Generate images for new scenes
	"""
	scenes = glob.glob(save_dir+sub_dirs[0]+'*')

	# file that saves the name
	if not os.path.exists(save_dir+'list/'):
		os.mkdir(save_dir+'list/')
	savefile = open(save_dir + 'list/all.txt','w')

	scenes_all = [\
		scene[-16::]
		for scene in scenes
	]
	for scene in scenes_all:
		savefile.write(scene+'\n')
			
	savefile.close()

	return 

if __name__ == '__main__':
	gen_dataset('phasor')
