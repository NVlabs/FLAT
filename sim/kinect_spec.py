# license:  Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
#           Licensed under the CC BY-NC-SA 4.0 license
#           (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode). 

# create kinect data spec
import numpy as np
from utils import *
import pdb
def kinect_sin_spec(cam):
	# camera function
	cam['T'] 		= np.array([2e4, 1e5, 4/3*1e4])
	cam['fr']		= np.array([\
		int(cam['T'][i]/cam['exp'])
		for i in range(len(cam['T'])) 
	])
	cam['tabs']	= np.array([\
		cam['dimt']
		for i in range(len(cam['T']))
	])
	cam['amp_e']	= [4.,4.,4.]
	cam['off_e']	= [0.,0.,0.]
	cam['amp_m']	= [2.,2.,2.]
	cam['off_m']	= [-1.,-1.,-1.]
	cam['phase'] 	= np.array([\
		[0, 2*PI/3, 4*PI/3]
		for i in range(len(cam['T']))
	])

	# create the camera function
	cam['cor'] = [\
		cam_sin_func(cam, i)\
		for i in range(len(cam['T']))
	]

	# mask
	cam['msk'] = kinect_mask()

	# cut off
	cam['raw_max'] = 0.01 # this brightness will be projected to 
	cam['map_max'] = 3000 # this brightness will be the threshold for the kinect output
	cam['lut_max'] = 3800 # this is the ending of lut table
	cam['sat'] = 32767 # this is the saturated brightness

	return cam

def cam_sin_func(cam, i):
	# precreate the camera function
	exp = cam['exp'] 
	# amplitude of lighting
	off_e = cam['off_e'][i]
	amp_e = cam['amp_e'][i]
	off_m = cam['off_m'][i]
	amp_m = cam['amp_m'][i]
	phase = cam['phase'][i]

	# constants
	tabs = cam['tabs'][i] # number of tabs/frs in impulse response
	fr = cam['fr'][i] # length of one period in frame number
	hf = (fr-1-1e-1)/2 # half of the period

	# create the signals
	idx = np.arange(fr)
	f = amp_e*np.sin(idx*2*PI/fr)+off_e
	g = np.stack(
			[\
				amp_m*(0.5-np.sign(np.mod(idx-fr*phase[i]/(2*PI),fr)-hf)/2)+off_m\
				for i in range(len(phase))
			],axis=1
		)
	
	# manually conpute the correlation
	cor = []
	for j in np.arange(tabs):
		cor.append(
			np.matmul(\
				np.expand_dims(
					amp_e*np.sin(np.mod(np.arange(fr)-j,fr)*2*PI/fr)+off_e,\
					axis=0
				),g
			)
		)
		
	cor = np.concatenate(cor,axis=0)

	# normalize 
	cor /= cam['T'][i]
	return cor

def kinect_mask():
	# return the kinect mask that creates the positive-negative interval
	mask = np.zeros((424,512))
	idx = 1
	for i in range(mask.shape[0]):
		mask[i,:] = idx
		if i != (mask.shape[0]/2-1):
			idx = -idx
	return mask

def kinect_real_spec(cam):
	# load the camera function
	prms = np.loadtxt('../params/kinect/cam_func_params.txt',delimiter=',')

	# camera function
	coef = 4*PI/3e-4
	cam['T'] 		= np.array([coef/prms[0], coef/prms[3], coef/prms[7]])
	cam['phase'] 	= -np.array([\
		[prms[1]*PI, (prms[1]+2/3)*PI, (prms[1]-2/3)*PI],
		[prms[4]*PI, (prms[4]+2/3)*PI, (prms[4]-2/3)*PI],
		[prms[8]*PI, (prms[8]+2/3)*PI, (prms[8]-2/3)*PI],
	])
	cam['A'] = np.array([prms[2], prms[6], prms[9]])
	cam['m'] = prms[5]

	cam['tabs']	= np.array([\
		cam['dimt']
		for i in range(len(cam['T']))
	])
	cam['t'] = (np.arange(cam['dimt']))*cam['exp']

	# create the camera function
	cam['cor'] = [\
		cam['A'][i]*np.sin(2*PI/cam['T'][i]*cam['t']-cam['phase'][i,j])
		for i in range(len(cam['T'])) for j in range(len(cam['phase'][i,:]))
	]
	for i in range(3,6):
		cam['cor'][i] = np.maximum(\
			np.minimum(cam['cor'][i],np.abs(cam['m'])),-np.abs(cam['m'])
		)
	cam['cor'] = np.array(cam['cor'])

	return cam

def kinect_real_tf_spec():
	# camera dict
	cam = {}
	cam['dimx'] = 512
	cam['dimy'] = 424
	cam['fov_x'] = 70

	# load the camera function
	prms = np.loadtxt('../params/kinect/cam_func_params.txt',delimiter=',')

	# camera function
	coef = 4*PI/3e-4
	cam['T'] 		= np.array([coef/prms[0], coef/prms[3], coef/prms[7]])
	cam['phase'] 	= -np.array([\
		[prms[1]*PI, (prms[1]+2/3)*PI, (prms[1]-2/3)*PI],
		[prms[4]*PI, (prms[4]+2/3)*PI, (prms[4]-2/3)*PI],
		[prms[8]*PI, (prms[8]+2/3)*PI, (prms[8]-2/3)*PI],
	])
	cam['A'] = np.array([prms[2], prms[6], prms[9]])
	cam['m'] = prms[5]
	return cam

def compute_cor(cam):
	# time frame
	cam['dimt'] += 20
	cam['tabs']	= np.array([\
		cam['dimt']
		for i in range(len(cam['T']))
	])
	cam['t'] = (np.arange(cam['dimt']))*cam['exp']

	# create the camera function
	cor = [\
		cam['A'][i]*np.sin(2*PI/cam['T'][i]*cam['t']-cam['phase'][i,j])
		for i in range(len(cam['T'])) for j in range(len(cam['phase'][i,:]))
	]
	for i in range(3,6):
		cor[i] = np.maximum(\
			np.minimum(cor[i],np.abs(cam['m'])),-np.abs(cam['m'])
		)
	return np.array(cor)

def nonlinear_adjust(cam, cor):
	cor = np.reshape(np.transpose(cor),[-1,3,3])
	phase = cam['phase']
	T = cam['T']
	depth_gt = cam['t']*C/2

	tmp_Qs = [[cor[:,k,j] * np.sin(phase[k][j]) for j in range(phase.shape[1])] for k in range(phase.shape[0])]
	tmp_Is = [[cor[:,k,j] * np.cos(phase[k][j]) for j in range(phase.shape[1])] for k in range(phase.shape[0])]
	tmp_Q = np.stack([np.sum(np.stack(tmp_Qs[k],-1),-1) for k in range(phase.shape[0])],-1)
	tmp_I = np.stack([np.sum(np.stack(tmp_Is[k],-1),-1) for k in range(phase.shape[0])],-1)
	ratio = tmp_Q / np.sqrt(tmp_Q**2 + tmp_I**2+1e-10)
	phase_pred = np.arctan2(tmp_Q, tmp_I)
	depth_pred = np.stack([phase_pred[:,k]*T[k]/2/PI * C/2 for k in range(3)],-1)

	# unwrapping manually			
	T_cri = T[1]
	depth_cri = depth_pred[:,1]
	depth_unwrap = []
	for k in range(3):
		unwrap = np.floor((depth_cri-depth_pred[:,k])/(T[k]*C/2)+0.5)
		depth_unwrap.append(depth_pred[:,k] + unwrap*(T[k]*C/2))
	depth_pred = np.stack(depth_unwrap,-1)

	cam['depth_atan'] = depth_pred
	cam['depth_true'] = depth_gt
	return cam

def kinect_mask():
	# return the kinect mask that creates the positive-negative interval
	mask = np.zeros((424,512))
	idx = 1
	for i in range(mask.shape[0]):
		mask[i,:] = idx
		if i != (mask.shape[0]/2-1):
			idx = -idx
	return mask
