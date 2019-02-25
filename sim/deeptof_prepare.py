# license:  Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
#           Licensed under the CC BY-NC-SA 4.0 license
#           (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode). 

# this code simulates the time-of-flight data of deeptof
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

def gen_raw(scene_n, data_dir, tof_cam):
	print('Processing scene', scene_n)

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

	# load the mask and classification
	with open('../FLAT/kinect_2/msk/'+scene_n[-23:-7],'rb') as f:
		msk_array=np.fromfile(f, dtype=np.float32)
	msk_array = np.reshape(msk_array,(cam['dimy'],cam['dimx'],4))
	msk = {}
	msk['background'] = msk_array[:,:,0]
	msk['edge'] = msk_array[:,:,1]
	msk['noise'] = msk_array[:,:,2]
	msk['reflection'] = msk_array[:,:,3]

	# compute mask
	msk_true_s = msk['background'] * msk['edge']
	depth_true_s = scipy.misc.imresize(depth_true,msk_array.shape[0:2],mode='F')

	# simulate the raw measurement
	res = tof_cam.process_gain_noise(cam, prop_idx, prop_s, scene, depth_true)
	meas = res['meas']
	dist = 3e-4*tof_cam.cam['T'][0]/4/PI*np.arctan2((meas[:,:,0]-meas[:,:,2]),(meas[:,:,1]-meas[:,:,3]))
	dist[np.where(dist < 0)] = 7.5 + dist[np.where(dist < 0)]
	depth = dist	

	err = np.sum(np.abs((depth-depth_true_s)*msk_true_s))/np.sum(msk_true_s)
	print('Mean err(m): ', err)

	depth_to_file = np.reshape(depth,-1).astype(np.float32)
	if not os.path.exists(data_dir):
		os.makedirs(data_dir)
	depth_to_file.tofile(data_dir+scene_n[-23:-7])

	return

def gen_dataset(setup):
	"""
	Create compact array from the dataset
	"""
	data_dir = '../FLAT/trans_render/static/'
	save_dir = '../FLAT/'+setup + '/'
	sub_dir = 'full/'

	if not os.path.exists(data_dir):
		os.mkdir(data_dir)

	if not os.path.exists(save_dir):
		os.mkdir(save_dir)
	if not os.path.exists(save_dir+sub_dir):
		os.mkdir(save_dir+sub_dir)

	tof_cam = eval(setup+'()')

	"""
	Create raw measurements from compact array
	"""
	# input the folder that contains the data
	scenes = glob.glob(data_dir+'*.pickle')

	# jump over those already finished 
	scenes_finished = glob.glob(save_dir+sub_dir+'*')
	scenes_finished = [scene[-16::] for scene in scenes_finished]

	for i in range(len(scenes)):
		if (scenes[i][-23:-7] in scenes) and (scenes[i][-23:-7] not in scenes_finished):
			gen_raw(scenes[i], data_dir, tof_cam)


	"""
	Generate images for new scenes
	"""
	scenes = glob.glob(save_dir+sub_dir+'*')

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
	gen_dataset('deeptof')
