# license:  Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
#           Licensed under the CC BY-NC-SA 4.0 license
#           (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode). 

# this code simulates the time-of-flight data of kinect
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
import sys

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
tf.logging.set_verbosity(tf.logging.INFO)

def gen_raw(scene_n, data_dir, tof_cam):
	print('Processing scene', scene_n)

	# four conditions
	lists = ['full','ideal','noise','reflection']
	funcs = [\
		tof_cam.process_delay_vig_gain_noise,
		tof_cam.process_gt_delay_vig_dist_surf_mapmax,
		tof_cam.process_gt_delay_vig_gain_noise,
		tof_cam.process_delay_vig_gain,
	]

	res = []
	# generate raw measurement
	for i in range(len(lists)):
		print(lists[i])
		# check if the file already exists
		goal_dir = data_dir+lists[i]+'/'
		if not os.path.isfile(goal_dir+scene_n[-23:-7]):
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

			# create raw measurements
			res.append(funcs[i](cam, prop_idx, prop_s, scene, depth_true))
			cam = deepcopy(tof_cam.cam)

			# write some measurements to binary file for the kinect to do it
			meas_to_file = res[-1]['meas']*tof_cam.cam['map_max']; # map back
			msk = kinect_mask()
			meas_to_file = meas_to_file*np.tile(np.expand_dims(msk,-1),[1,1,9])
			meas_to_file = np.reshape(meas_to_file,-1).astype(np.int32)
			if not os.path.exists(goal_dir):
				os.makedirs(goal_dir)
			meas_to_file.tofile(goal_dir+scene_n[-23:-7])

	return

def gen_gt(scene_n, data_dir, tof_cam):
	print('Processing ground truth scene', scene_n)

	# check if the file already exists
	goal_dir = data_dir+'gt/'
	if not os.path.isfile(goal_dir+scene_n[-23:-7]):
		with open(scene_n,'rb') as f:
			data = pickle.load(f)

		# save the ground truth
		depth_true = data['depth_true']
		depth_true = np.reshape(depth_true,-1).astype(np.float32)
		if not os.path.exists(goal_dir):
			os.makedirs(goal_dir)
		depth_true.tofile(goal_dir+scene_n[-23:-7])

	return

def gen_raw_dyn(scene_n, data_dir, tof_cam):
	print('Processing scene', scene_n)

	# four conditions
	lists = ['full','ideal','noise','reflection']
	funcs = [\
		tof_cam.process_delay_vig_gain_noise,
		tof_cam.process_gt_delay_vig_dist_surf_mapmax,
		tof_cam.process_gt_delay_vig_gain_noise,
		tof_cam.process_delay_vig_gain,
	]

	res = []
	mid = 4
	# generate raw measurement
	for i in range(len(lists)):
		print(lists[i])
		# check if the file already exists
		goal_dir = data_dir+lists[i]+'/'
		if not os.path.isfile(goal_dir+scene_n[-23:-7]):
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

			if lists[i]=='full':
				meas = []
				for j in range(len(prop_idx)):
					res.append(funcs[i](cam, prop_idx[j], prop_s[j], scene, depth_true))
					cam = deepcopy(tof_cam.cam)
					meas.append(res[-1]['meas'][:,:,j])
				meas = np.stack(meas, -1)
			else:
				res.append(funcs[i](cam, prop_idx[mid], prop_s[mid], scene, depth_true))
				cam = deepcopy(tof_cam.cam)
				meas = res[-1]['meas']

			# write some measurements to binary file for the kinect to do it
			meas_to_file = meas*tof_cam.cam['map_max']; # map back
			msk = kinect_mask()
			meas_to_file = meas_to_file*np.tile(np.expand_dims(msk,-1),[1,1,9])
			meas_to_file = np.reshape(meas_to_file,-1).astype(np.int32)
			if not os.path.exists(goal_dir):
				os.makedirs(goal_dir)
			meas_to_file.tofile(goal_dir+scene_n[-23:-7])

	return

def gen_gt_dyn(scene_n, data_dir, tof_cam):
	print('Processing ground truth scene', scene_n)

	# check if the file already exists
	goal_dir = data_dir+'gt/'
	if not os.path.isfile(goal_dir+scene_n[-23:-7]):
		with open(scene_n,'rb') as f:
			data = pickle.load(f)

		# save the ground truth
		depth_true = data['depth_true']
		depth_true = np.reshape(depth_true,-1).astype(np.float32)
		if not os.path.exists(goal_dir):
			os.makedirs(goal_dir)
		depth_true.tofile(goal_dir+scene_n[-23:-7])

	return

def metric_valid(depth, gt, msk):
	# compute mean absolute error on places where msk = 1
	msk /= np.sum(msk)
	return np.sum(np.abs(depth - gt)*msk)

def data_augment_full(scene_n, test_dir, tof_cam):
	print('Augmenting scene', scene_n)

	# generate raw measurement
	cam = tof_cam.cam
	with open(test_dir+'full/'+scene_n[-16::],'rb') as f:
		meas=np.fromfile(f, dtype=np.int32)
	meas = np.reshape(meas,(cam['dimy'],cam['dimx'],9)).astype(np.float32)
	msk = kinect_mask().astype(np.float32)
	meas_gt = meas

	# reduce the resolution of the depth
	depth_true_s = np.zeros(meas.shape[0:2])
	msk_true_s = (np.abs(depth_true_s) > 1e-4)
	true = np.stack([depth_true_s,msk_true_s],2)
	true = np.concatenate([true, meas_gt], 2)

	# the input of the network
	return meas, true

def data_augment_ideal(scene_n, test_dir, tof_cam):
	print('Augmenting scene', scene_n)

	# generate raw measurement
	cam = tof_cam.cam
	with open(test_dir+'ideal/'+scene_n[-16::],'rb') as f:
		meas=np.fromfile(f, dtype=np.int32)
	meas = np.reshape(meas,(cam['dimy'],cam['dimx'],9)).astype(np.float32)
	msk = kinect_mask().astype(np.float32)
	meas_gt = meas

	# reduce the resolution of the depth
	depth_true_s = np.zeros(meas.shape[0:2])
	msk_true_s = (np.abs(depth_true_s) > 1e-4)
	true = np.stack([depth_true_s,msk_true_s],2)
	true = np.concatenate([true, meas_gt], 2)

	# the input of the network
	return meas, true

def data_augment_noise(scene_n, test_dir, tof_cam):
	print('Augmenting scene', scene_n)

	# generate raw measurement
	cam = tof_cam.cam
	with open(test_dir+'noise/'+scene_n[-16::],'rb') as f:
		meas=np.fromfile(f, dtype=np.int32)
	meas = np.reshape(meas,(cam['dimy'],cam['dimx'],9)).astype(np.float32)
	msk = kinect_mask().astype(np.float32)
	meas_gt = meas

	# reduce the resolution of the depth
	depth_true_s = np.zeros(meas.shape[0:2])
	msk_true_s = (np.abs(depth_true_s) > 1e-4)
	true = np.stack([depth_true_s,msk_true_s],2)
	true = np.concatenate([true, meas_gt], 2)

	# the input of the network
	return meas, true

def data_augment_reflection(scene_n, test_dir, tof_cam):
	print('Augmenting scene', scene_n)

	# generate raw measurement
	cam = tof_cam.cam
	with open(test_dir+'reflection/'+scene_n[-16::],'rb') as f:
		meas=np.fromfile(f, dtype=np.int32)
	meas = np.reshape(meas,(cam['dimy'],cam['dimx'],9)).astype(np.float32)
	msk = kinect_mask().astype(np.float32)
	meas_gt = meas

	# reduce the resolution of the depth
	depth_true_s = np.zeros(meas.shape[0:2])
	msk_true_s = (np.abs(depth_true_s) > 1e-4)
	true = np.stack([depth_true_s,msk_true_s],2)
	true = np.concatenate([true, meas_gt], 2)

	# the input of the network
	return meas, true

def pixel_class(depth, gt):
	# classification of pixels
	msk = {}
	
	# background mask 
	msk['background'] = np.ones(gt.shape)
	msk['background'][np.where(gt<1e-4)] = 0

	# edge mask
	err_ideal = np.abs(depth['ideal']-gt)
	msk['edge'] = np.ones(gt.shape)
	msk['edge'][np.where(err_ideal>0.05)] = 0

	# noise 
	msk['noise'] = (depth['ideal'] - depth['noise'])*msk['edge']*msk['background']


	# reflection
	msk['reflection'] = (depth['ideal'] - depth['reflection'])*msk['edge']*msk['background']

	return msk

def testing_msk(tests, test_dir, tof_cam, tof_net, base_cor, cam):
	# testing
	errs = []
	errs_base = []
	errs_num_pix = []
	errs_total = []
	errs_base_total = []   

	goal_dir = test_dir+'msk/'
	if not os.path.exists(goal_dir):
		os.makedirs(goal_dir) 

	for iter_idx in range(len(tests)):
		if not os.path.isfile(goal_dir+tests[iter_idx][-16::]):
			with open(test_dir+'gt/'+tests[iter_idx][-16::],'rb') as f:
				gt=np.fromfile(f, dtype=np.float32)
			depth_true = np.reshape(gt,(cam['dimy']*4,cam['dimx']*4))

			# reduce the resolution of the depth
			depth_true_s = scipy.misc.imresize(\
				depth_true,\
				[cam['dimy'],cam['dimx']],\
				mode='F'\
			)
			# load ground truth, convert distance to depth
			depth_true = tof_cam.dist_to_depth(depth_true_s)

			# elaborate the camera
			tof_cam.cam['dimx'] = cam['dimx']
			tof_cam.cam['dimy'] = cam['dimy']
			tof_cam.cam['dimt'] = cam['dimt']
			tof_cam.cam['exp'] = cam['exp']
			cor = compute_cor(tof_cam.cam)
			cam = nonlinear_adjust(tof_cam.cam, cor)

			depths = {}
			trues = {}
			lists = ['full','ideal','noise','reflection']
			key_idx = 1
			for key in lists:
				# input network
				x = []
				y = []
				x_te, y_te = eval('data_augment_'+key+'(tests[iter_idx], test_dir, tof_cam)')
				x.append(x_te)
				y.append(y_te)
				x = np.stack(x,0)
				y = np.stack(y,0)
	
				# evaluate the model and print results
				eval_results = tof_net.evaluate(x=x,y=y)

				# predict data
				res = list(tof_net.predict(x=x))[0]

				# save the depth
				depths[key] = res['depth']
	
			# classify the pixels
			msk = pixel_class(depths, depth_true)

			msk_array = np.stack([msk['background'],msk['edge'],msk['noise'],msk['reflection']],-1)
			msk_array = msk_array.astype(np.float32)
			msk_array.tofile(goal_dir+tests[iter_idx][-16::])

	return

def testing(tests, test_dir, output_dir, tof_cam, tof_net, base_cor, cam):
	# testing
	errs = []
	errs_base = []
	errs_num_pix = []
	errs_total = []
	errs_base_total = []	

	for iter_idx in range(len(tests)):
		if not os.path.isfile(output_dir+tests[iter_idx][-16::]+'.png'):
			with open(test_dir+'gt/'+tests[iter_idx][-16::],'rb') as f:
				gt=np.fromfile(f, dtype=np.float32)
			depth_true = np.reshape(gt,(cam['dimy']*4,cam['dimx']*4))
			
			# reduce the resolution of the depth
			depth_true_s = scipy.misc.imresize(\
				depth_true,\
				[cam['dimy'],cam['dimx']],\
				mode='F'\
			)

			# elaborate the camera
			tof_cam.cam['dimx'] = cam['dimx']
			tof_cam.cam['dimy'] = cam['dimy']
			tof_cam.cam['dimt'] = cam['dimt']
			tof_cam.cam['exp'] = cam['exp']
			cor = compute_cor(tof_cam.cam)
			cam = nonlinear_adjust(tof_cam.cam, cor)

			depths = {}
			trues = {}
			lists = ['full','ideal','noise','reflection']
			key_idx = 1
			for key in lists:
				# input network
				x = []
				y = []
				x_te, y_te = eval('data_augment_'+key+'(tests[iter_idx], test_dir, tof_cam)')
				x.append(x_te)
				y.append(y_te)
				x = np.stack(x,0)
				y = np.stack(y,0)

				# evaluate the model and print results
				eval_results = tof_net.evaluate(x=x,y=y)

				# predict data
				res = list(tof_net.predict(x=x))[0]

				# save the depth
				depths[key] = res['depth']

			# load ground truth, convert distance to depth
			depth_true = tof_cam.dist_to_depth(depth_true_s)

			# load the mask and classification
			with open(test_dir+'msk/'+tests[iter_idx][-16::],'rb') as f:
				msk_array=np.fromfile(f, dtype=np.float32)
			msk_array = np.reshape(msk_array,(cam['dimy'],cam['dimx'],4))
			msk = {}
			msk['background'] = msk_array[:,:,0]
			msk['edge'] = msk_array[:,:,1]
			msk['noise'] = msk_array[:,:,2]
			msk['reflection'] = msk_array[:,:,3]

			# compute mask
			msk_gt = msk['background'] * msk['edge'] 

			fig=plt.figure(figsize=(16,7))
			for key in lists:
				# the true depth
				depth = depths[key]
				depth[np.where(np.isnan(depth))] = 0

				# compute mask
				msk_base_gt = deepcopy(msk_gt)
				msk_base_gt[np.where(depth<1e-4)] = 0		  

				# visualization
				vmin = (depth*msk_gt).min()
				vmax = (depth*msk_gt).max()
				emin = -.1
				emax = .1

				ax=fig.add_subplot(2,4,key_idx)
				plt.imshow(depth*msk_gt,vmin=vmin, vmax=vmax)
				plt.axis('off')
				plt.title(key)
				plt.colorbar()

				ax=fig.add_subplot(2,4,key_idx+1)
				plt.imshow((depth-depth_true)*(msk_base_gt>0),vmin=emin,vmax=emax)
				plt.axis('off')
				err_base=metric_valid(depth,depth_true,msk_base_gt)
				plt.title('Base err: '+'%.06f'%err_base+'m')
				plt.colorbar()

				key_idx += 2
			
			plt.savefig(\
				output_dir+tests[iter_idx][-16::]+'.png',
				bbox_inches='tight',
				dpi = 600,
			)
			plt.close('all')

	return

if __name__ == '__main__':
	"""
	Create compact array from the dataset
	"""
	setup = 'kinect'
	data_dir = '../FLAT/trans_render/static/'
	data_dyn_dir = '../FLAT/trans_render/dyn/'
	goal_dir = '../FLAT/'+setup + '/'
	prms_dir = "../params/"+setup+'/'
	models_dir = "../pipe/models/kinect/"
	sys.path.insert(0,'../pipe/')
	sys.path.insert(0,'../params/kinect/')

	if not os.path.exists(goal_dir):
		os.makedirs(goal_dir)

	# load the camera
	scenes = glob.glob(data_dir+'*.pickle')
	with open(scenes[0],'rb') as f:
		data = pickle.load(f)
	cam = data['cam']

	"""
	Create four raw measurements from compact array
	"""
	# input the folder that contains the data
	scenes = glob.glob(data_dir+'*.pickle')

	# initialize the camera model
	tof_cam = kinect_real_tf()
	for i in range(len(scenes)):
		gen_raw(scenes[i], goal_dir, tof_cam)


	"""
	Generate ground truth data
	"""
	for i in range(len(scenes)):
		gen_gt(scenes[i], goal_dir, tof_cam)


	"""
	Create four raw measurements from compact array - dynamic
	"""
	# input the folder that contains the data
	scenes = glob.glob(data_dyn_dir+'*.pickle')

	# initialize the camera model
	tof_cam = kinect_real_tf()
	for i in range(len(scenes)):
		gen_raw_dyn(scenes[i], goal_dir, tof_cam)


	"""
	Generate ground truth data
	"""
	for i in range(len(scenes)):
		gen_gt_dyn(scenes[i], goal_dir, tof_cam)


	"""
	Generate masks
	"""
	data_dir = goal_dir+'full/'
	scenes = glob.glob(data_dir+'*')
	file_name = 'LF2'
	from LF2 import tof_net_func
	tof_net = learn.Estimator(
		model_fn=tof_net_func,
		model_dir=models_dir+file_name,
	)

	# read the baseline corrector
	with open(prms_dir+'kinect_baseline_correct.pickle','rb') as f:
		base_cor = pickle.load(f)

	testing_msk(scenes, goal_dir, tof_cam, tof_net, base_cor, cam)


	"""
	Generate images for new scenes
	"""
	scenes = glob.glob(data_dir+'*')

	# create the network estimator
	file_name = 'LF2'
	from LF2 import tof_net_func
	tof_net = learn.Estimator(
		model_fn=tof_net_func,
		model_dir=models_dir+file_name,
	)

	# create output folder
	output_dir = goal_dir + 'list/'
	folder_name = 'all'	
	output_dir = output_dir + folder_name + '/'
	if not os.path.exists(output_dir):
		os.mkdir(output_dir)

	# read the baseline corrector
	with open(prms_dir+'kinect_baseline_correct.pickle','rb') as f:
		base_cor = pickle.load(f)

	testing(scenes, goal_dir, output_dir, tof_cam, tof_net, base_cor, cam)

	# file that saves the name
	savefile = open(goal_dir + 'list/all.txt','w')

	scenes_all = [\
		scene[-16::]
		for scene in scenes
	]
	for scene in scenes_all:
		savefile.write(scene+'\n')
			
	savefile.close()


	"""
	Put the images into correct folder based on the txt
	"""
	folders = [\
		'test',\
		'val',\
		'train',\
		'shot_noise_test',
		'test_dyn',
		'motion_background',
		'motion_foreground',
		'motion_real'
	]
	for folder in folders:
		if os.path.exists(goal_dir+'list/'+folder+'.txt'):
			f = open(goal_dir+'list/'+folder+'.txt','r')
			message = f.read()
			files = message.split('\n')
			files = files[0:-1]
			files = [file for file in files]
			if not os.path.exists(goal_dir+'list/'+folder):
				os.mkdir(goal_dir+'list/'+folder)

			for i in range(len(files)):
				copyfile(\
					goal_dir+'list/all/'+files[i]+'.png', 
					goal_dir+'list/'+folder+'/'+files[i]+'.png'
				)



	"""
	Update the txt based on the images
	"""
	for folder in folders:
		scenes = glob.glob(goal_dir+'list/'+folder+'/'+'*.png')
		savefile = open(goal_dir+'list/'+folder+'.txt','w')
		for scene in scenes:
			filename = scene[-20:-4]
			savefile.write(filename+'\n')
		savefile.close()

	"""
	Sanity check
	"""
	f_trains = open(goal_dir+'list/train.txt','r')
	f_vals = open(goal_dir+'list/val.txt','r')
	f_tests = open(goal_dir+'list/test.txt','r')

	flg_vals = np.array([f_val in f_trains for f_val in f_vals],np.float32)
	flg_tests = np.array([f_test in f_trains for f_test in f_tests],np.float32)
	flg_vals = np.sum(flg_vals)
	flg_tests = np.sum(flg_tests)

	if flg_vals or flg_tests:
		print("Sanity check failed!!")
	else:
		print("Congratulations! Sanity check passed.")

