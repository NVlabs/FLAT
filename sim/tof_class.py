# license:  Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
#           Licensed under the CC BY-NC-SA 4.0 license
#           (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode). 

# this file describes lots of different camera classes
import numpy as np
import pdb
import tensorflow as tf
from kinect_spec import *
from utils import *
import scipy.misc
import scipy.interpolate
from matplotlib import pyplot as plt

class cam_baseline:
	# baseline tof camera, uses square wave for emission and modulation
	# other cameras can inherit from this class
	def __init__(self,cam):
		for key in cam.keys():
			self.cam[key] = cam[key]

		# create the camera function
		self.cam['T'] 		= 1e6
		self.cam['fr']		= self.cam['T']/self.cam['exp']
		self.cam['tabs']	= self.cam['dimt']
		self.cam['amp_e']	= 4.
		self.cam['off_e']	= 2.
		self.cam['amp_m']	= 1.
		self.cam['off_m']	= 1.
		self.cam['phase'] 	= np.array([0, PI/2, PI, 3*PI/2])

		self.vars = {}
		self.dtype= tf.float32
		# build computation graph
		self.graph = tf.Graph()
		self.session = tf.Session(graph = self.graph)
		self.build_graph()

	def build_graph(self):
		# shorten the name
		cam = self.cam
		with self.graph.as_default():
			# inputs
			# impulse response
			ipr_init = np.zeros((cam['dimy'],cam['dimx'],cam['dimt']))
			self.ipr_in = tf.Variable(ipr_init,dtype=self.dtype)
			ipr = tf.Variable(self.ipr_in,dtype=self.dtype) 
			# variables
			# exposure time of each frame
			exp = tf.Variable(cam['exp'],dtype=self.dtype) 
			# amplitude of lighting
			amp_e = tf.Variable(cam['amp_e'],dtype=self.dtype) 
			# offset of lighting
			off_e = tf.Variable(cam['off_e'],dtype=self.dtype)
			# amplitude of modulation
			amp_m = tf.Variable(cam['amp_m'],dtype=self.dtype) 
			# offset of modulation
			off_m = tf.Variable(cam['off_m'],dtype=self.dtype)

			# constants
			tabs = cam['tabs'] # number of tabs/frs in impulse response
			fr = cam['fr'] # length of one period in frame number
			hf = (fr-1-1e-1)/2 # half of the period

			# create the signals
			idx = tf.constant(np.arange(fr),dtype=self.dtype)
			f = amp_e*(0.5-tf.sign(idx-hf)/2)+off_e
			g = tf.stack(
					[\
						amp_m*(0.5-tf.sign(idx-hf)/2)+off_m,
						amp_m*(0.5-tf.sign(np.mod(idx-fr/2,fr)-hf)/2)+off_m,
						amp_m*(0.5-tf.sign(np.mod(idx-fr/4,fr)-hf)/2)+off_m,
						amp_m*(0.5-tf.sign(np.mod(idx-3*fr/4,fr)-hf)/2)+off_m,
					],
					axis=1
				)

			# manually conduct partial correlation
			with tf.device('/cpu:0'):
			# the partial correlation needs too large memory to GPU
			# so we use CPU instead
				cor = []
				for i in np.arange(tabs):
					cor.append(
						tf.matmul(\
							tf.expand_dims(
								amp_e*(0.5-tf.sign(\
									tf.constant(\
										np.mod(np.arange(fr)-i,fr)-hf,\
										dtype=self.dtype\
									)
								)/2)+off_e,\
							axis=0),g
						)
					)
			cor = tf.concat(cor,axis=0)			

			# compute the raw measurement 
			cor_exp = tf.tile(\
				tf.expand_dims(tf.expand_dims(cor,0),0),
				[cam['dimy'],cam['dimx'],1,1]
			)
			ipr_exp = tf.tile(\
				tf.expand_dims(ipr,-1),[1,1,1,4]
			)
			meas = tf.reduce_sum(cor_exp * ipr_exp, 2)

			# phase and depth
			phase = \
				(meas[:,:,2]-meas[:,:,3])/2/\
				(\
					tf.abs(meas[:,:,0]-meas[:,:,1])+\
					tf.abs(meas[:,:,2]-meas[:,:,3])\
				)
			depth = phase * hf * exp * C /2

			# save some data for debugging
			self.vars['ipr'] = ipr
			self.vars['g'] = g
			self.vars['f'] = f
			self.vars['cor'] = cor
			self.vars['meas'] = meas
			self.vars['phase'] = phase
			self.vars['depth']	= depth

			# input data
			self.input_data = tf.group(\
				ipr.assign(self.ipr_in)
			)

			# 
			init_op = tf.global_variables_initializer()
			self.session.run(init_op)
		return

	def process(self,prop):
		# process data
		self.input_dict = {
			self.ipr_in		:		prop,
		}
		self.session.run(self.input_data, self.input_dict)
		res_dict = {
			'f'				:		self.vars['f'],
			'g'				:		self.vars['g'],
			'cor'			:		self.vars['cor'],
			'meas'			:		self.vars['meas'],
			'phase'			:		self.vars['phase'],
			'depth'			:		self.vars['depth'],
		}
		return self.session.run(res_dict)

class cam_sin(cam_baseline):
	# baseline tof camera, uses square wave for emission and modulation
	# other cameras can inherit from this class
	def __init__(self,cam):
		for key in cam.keys():
			self.cam[key] = cam[key]

		# create the camera function
		self.cam['T'] 		= 5e4
		self.cam['fr']		= self.cam['T']/self.cam['exp']
		self.cam['tabs']	= self.cam['dimt']
		self.cam['amp_e']	= 1.
		self.cam['off_e']	= 1.
		self.cam['amp_m']	= 1.
		self.cam['off_m']	= 1.
		self.cam['phase'] 	= np.array([0, PI/2, PI, 3*PI/2])

		self.vars = {}
		self.dtype= tf.float32
		# build computation graph
		self.graph = tf.Graph()
		self.session = tf.Session(graph = self.graph)
		self.build_graph()

	def build_graph(self):
		# shorten the name
		cam = self.cam
		with self.graph.as_default():
			# inputs
			# impulse response
			ipr_init = np.zeros((cam['dimy'],cam['dimx'],cam['dimt']))
			self.ipr_in = tf.Variable(ipr_init,dtype=self.dtype)
			ipr = tf.Variable(self.ipr_in,dtype=self.dtype) 
			# variables
			# exposure time of each frame
			exp = tf.Variable(cam['exp'],dtype=self.dtype) 
			# amplitude of lighting
			off_e = tf.Variable(cam['off_e'],dtype=self.dtype)
			amp_e = tf.Variable(cam['amp_e'],dtype=self.dtype) 
			off_m = tf.Variable(cam['off_m'],dtype=self.dtype)
			amp_m = tf.Variable(cam['amp_m'],dtype=self.dtype) 

			# constants
			tabs = cam['tabs'] # number of tabs/frs in impulse response
			fr = cam['fr'] # length of one period in frame number
			hf = (fr-1-1e-1)/2 # half of the period

			# create the signals
			idx = tf.constant(np.arange(fr),dtype=self.dtype)
			f = amp_e*tf.sin(idx*2*PI/fr)+off_e
			g = tf.stack(
					[\
						amp_m*tf.sin(idx*2*PI/fr)+off_m,
						amp_m*tf.sin(np.mod(idx-fr/2,fr)*2*PI/fr)+off_m,
						amp_m*tf.sin(np.mod(idx-fr/4,fr)*2*PI/fr)+off_m,
						amp_m*tf.sin(np.mod(idx-3*fr/4,fr)*2*PI/fr)+off_m,
					],
					axis=1
				)

			# manually conduct partial correlation
			with tf.device('/cpu:0'):
			# the partial correlation needs too large memory to GPU
			# so we use CPU instead
				cor = []
				for i in np.arange(tabs):
					cor.append(
						tf.matmul(\
							tf.expand_dims(
								amp_e*np.sin(np.mod(np.arange(fr)-i,fr)*2*PI/fr)+off_e,\
								axis=0
							),g
						)
					)
			cor = tf.concat(cor,axis=0)			

			# compute the raw measurement 
			cor_exp = tf.tile(\
				tf.expand_dims(tf.expand_dims(cor,0),0),
				[cam['dimy'],cam['dimx'],1,1]
			)
			ipr_exp = tf.tile(\
				tf.expand_dims(ipr,-1),[1,1,1,4]
			)
			meas = tf.reduce_sum(cor_exp * ipr_exp, 2)

			# phase and depth
			phase = tf.atan((meas[:,:,2]-meas[:,:,3])/(meas[:,:,0]-meas[:,:,1]))
			ampl = tf.sqrt((meas[:,:,2]-meas[:,:,3])**2+(meas[:,:,0]-meas[:,:,1])**2)
			depth = phase * cam['T']/2/PI * C /2

			# save some data for debugging
			self.vars['ipr'] = ipr
			self.vars['g'] = g
			self.vars['f'] = f
			self.vars['cor'] = cor
			self.vars['meas'] = meas
			self.vars['phase'] = phase
			self.vars['ampl'] = ampl
			self.vars['depth']	= depth

			# input data
			self.input_data = tf.group(\
				ipr.assign(self.ipr_in)
			)

			# 
			init_op = tf.global_variables_initializer()
			self.session.run(init_op)
		return

	def process(self,ipr_idx,ipr_s):
		depth = self.sess.run(\
				self.vars['depth'],
				feed_dict={\
					self.vars['ipr_s']:ipr_s,
				}
			)


		return depth

class cam_real(cam_baseline):
	# baseline tof camera, uses square wave for emission and modulation
	# other cameras can inherit from this class
	def __init__(self,cam):
		for key in cam.keys():
			self.cam[key] = cam[key]

		# create the camera function
		self.cam['T'] 		= 1e6
		self.cam['fr']		= self.cam['T']/self.cam['exp']
		self.cam['tabs']	= self.cam['dimt']
		self.cam['amp_e']	= 4.
		self.cam['off_e']	= 2.
		self.cam['amp_m']	= 1.
		self.cam['off_m']	= 1.
		self.cam['phase'] 	= np.array([0, PI/2, PI, 3*PI/2])

		self.vars = {}
		self.dtype= tf.float32
		# build computation graph
		self.graph = tf.Graph()
		self.session = tf.Session(graph = self.graph)
		self.build_graph()

	def build_graph(self):
		# shorten the name
		cam = self.cam
		with self.graph.as_default():
			# inputs
			# impulse response
			ipr_init = np.zeros((cam['dimy'],cam['dimx'],cam['dimt']))
			self.ipr_in = tf.Variable(ipr_init,dtype=self.dtype)
			ipr = tf.Variable(self.ipr_in,dtype=self.dtype) 
			# variables
			# exposure time of each frame
			exp = tf.Variable(cam['exp'],dtype=self.dtype) 
			# amplitude of lighting
			off_e = tf.Variable(cam['off_e'],dtype=self.dtype)
			amp_e = tf.Variable(cam['amp_e'],dtype=self.dtype) 
			off_m = tf.Variable(cam['off_m'],dtype=self.dtype)
			amp_m = tf.Variable(cam['amp_m'],dtype=self.dtype) 

			# constants
			tabs = cam['tabs'] # number of tabs/frs in impulse response
			fr = cam['fr'] # length of one period in frame number
			hf = (fr-1-1e-1)/2 # half of the period

			# create the signals
			idx = tf.constant(np.arange(fr),dtype=self.dtype)
			f = amp_e*tf.sin(idx*2*PI/fr)+off_e
			g = tf.stack(
					[\
						amp_m*(0.5-tf.sign(idx-hf)/2)+off_m,
						amp_m*(0.5-tf.sign(np.mod(idx-fr/2,fr)-hf)/2)+off_m,
						amp_m*(0.5-tf.sign(np.mod(idx-fr/4,fr)-hf)/2)+off_m,
						amp_m*(0.5-tf.sign(np.mod(idx-3*fr/4,fr)-hf)/2)+off_m,
					],
					axis=1
				)
			
			# manually conduct partial correlation
			with tf.device('/cpu:0'):
			# the partial correlation needs too large memory to GPU
			# so we use CPU instead
				cor = []
				for i in np.arange(tabs):
					cor.append(
						tf.matmul(\
							tf.expand_dims(
								amp_e*np.sin(np.mod(np.arange(fr)-i,fr)*2*PI/fr)+off_e,\
								axis=0
							),g
						)
					)
			cor = tf.concat(cor,axis=0)			

			# compute the raw measurement 
			cor_exp = tf.tile(\
				tf.expand_dims(tf.expand_dims(cor,0),0),
				[cam['dimy'],cam['dimx'],1,1]
			)
			ipr_exp = tf.tile(\
				tf.expand_dims(ipr,-1),[1,1,1,4]
			)
			meas = tf.reduce_sum(cor_exp * ipr_exp, 2)

			# phase and depth
			phase = tf.atan((meas[:,:,2]-meas[:,:,3])/(meas[:,:,0]-meas[:,:,1]))
			ampl = tf.sqrt((meas[:,:,2]-meas[:,:,3])**2+(meas[:,:,0]-meas[:,:,1])**2)
			depth = phase * cam['T']/2/PI * C /2

			# save some data for debugging
			self.vars['ipr'] = ipr
			self.vars['g'] = g
			self.vars['f'] = f
			self.vars['cor'] = cor
			self.vars['meas'] = meas
			self.vars['phase'] = phase
			self.vars['ampl'] = ampl
			self.vars['depth']	= depth

			# input data
			self.input_data = tf.group(\
				ipr.assign(self.ipr_in)
			)

			# 
			init_op = tf.global_variables_initializer()
			self.session.run(init_op)
		return

class cam_real_fast(cam_baseline):
	# baseline tof camera, uses square wave for emission and modulation
	# other cameras can inherit from this class
	def __init__(self,cam):
		self.cam = {}
		for key in cam.keys():
			self.cam[key] = cam[key]

		# create the camera function
		self.cam['T'] 		= 1e6
		self.cam['fr']		= self.cam['T']/self.cam['exp']
		self.cam['tabs']	= self.cam['dimt']
		self.cam['amp_e']	= 4.
		self.cam['off_e']	= 2.
		self.cam['amp_m']	= 1.
		self.cam['off_m']	= 1.
		self.cam['phase'] 	= np.array([0, PI/2, PI, 3*PI/2])

		# create the camera function
		self.cor = self.cam_func()

		self.vars = {}
		self.dtype= tf.float32
		# build computation graph
		self.graph = tf.Graph()
		self.session = tf.Session(graph = self.graph)
		self.build_graph()

	def cam_func(self):
		# precreate the camera function
		exp = self.cam['exp'] 
		# amplitude of lighting
		off_e = self.cam['off_e']
		amp_e = self.cam['amp_e'] 
		off_m = self.cam['off_m']
		amp_m = self.cam['amp_m'] 
		phase = self.cam['phase']

		# constants
		tabs = self.cam['tabs'] # number of tabs/frs in impulse response
		fr = self.cam['fr'] # length of one period in frame number
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
		for i in np.arange(tabs):
			cor.append(
				np.matmul(\
					np.expand_dims(
						amp_e*np.sin(np.mod(np.arange(fr)-i,fr)*2*PI/fr)+off_e,\
						axis=0
					),g
				)
			)
		cor = np.concatenate(cor,axis=0)
		return cor

	def build_graph(self):
		# shorten the name
		cam = self.cam
		with self.graph.as_default():
			# inputs
			# impulse response
			ipr_init = np.zeros((cam['dimy'],cam['dimx'],cam['dimt']))
			self.ipr_in = tf.Variable(ipr_init,dtype=self.dtype)
			ipr = tf.Variable(self.ipr_in,dtype=self.dtype) 
			# camera function
			cor = tf.constant(self.cor, dtype=self.dtype)

			# compute the raw measurement 
			cor_exp = tf.tile(\
				tf.expand_dims(tf.expand_dims(cor,0),0),
				[cam['dimy'],cam['dimx'],1,1]
			)
			ipr_exp = tf.tile(\
				tf.expand_dims(ipr,-1),[1,1,1,4]
			)
			meas = tf.reduce_sum(cor_exp * ipr_exp, 2)

			# phase and depth
			Q = tf.zeros((cam['dimy'], cam['dimx']))
			I = tf.zeros((cam['dimy'], cam['dimx']))
			for i in range(len(cam['phase'])):
				Q += meas[:,:,i] * tf.sin(cam['phase'][i].astype(np.float32))
				I += meas[:,:,i] * np.cos(cam['phase'][i].astype(np.float32))

			# the generalized form of phase stepping
			phase = tf.atan(Q/I)
			ampl = 2*tf.sqrt(Q**2+I**2)/len(cam['phase'])
			depth = phase * cam['T']/2/PI * C /2

			# save some data for debugging
			self.vars['ipr'] = ipr
			self.vars['cor'] = cor
			self.vars['meas'] = meas
			self.vars['phase'] = phase
			self.vars['ampl'] = ampl
			self.vars['depth']	= depth # unwrapped depth

			# input data
			self.input_data = tf.group(\
				ipr.assign(self.ipr_in)
			)

			# 
			init_op = tf.global_variables_initializer()
			self.session.run(init_op)
		return

	def process(self,prop):
		# process data
		self.input_dict = {
			self.ipr_in		:		prop,
		}
		self.session.run(self.input_data, self.input_dict)
		res_dict = {
			'meas'			:		self.vars['meas'],
			'phase'			:		self.vars['phase'],
			'ampl'			:		self.vars['ampl'],
		}
		# phase unwrapping
		result = self.session.run(res_dict)
		result['phase'][np.where(result['phase']<0)] += PI
		result['depth'] = result['phase'] * self.cam['T']/2/PI * C /2
		return result

class cam_real_np(cam_baseline):
	# baseline tof camera, uses square wave for emission and modulation
	# other cameras can inherit from this class
	def __init__(self,cam):
		for key in cam.keys():
			self.cam[key] = cam[key]

		# create the camera function
		self.cam['T'] 		= 1e6
		self.cam['fr']		= self.cam['T']/self.cam['exp']
		self.cam['tabs']	= self.cam['dimt']
		self.cam['amp_e']	= 4.
		self.cam['off_e']	= 2.
		self.cam['amp_m']	= 1.
		self.cam['off_m']	= 1.
		self.cam['phase'] 	= np.array([0, PI/2, PI, 3*PI/2])

		# create the camera function
		self.cor = self.cam_func()

		self.vars = {}
		self.dtype= tf.float32

	def cam_func(self):
		# precreate the camera function
		exp = self.cam['exp'] 
		phase = self.cam['phase']
		# amplitude of lighting
		off_e = self.cam['off_e']
		amp_e = self.cam['amp_e'] 
		off_m = self.cam['off_m']
		amp_m = self.cam['amp_m'] 

		# constants
		tabs = self.cam['tabs'] # number of tabs/frs in impulse response
		fr = self.cam['fr'] # length of one period in frame number

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
		for i in np.arange(tabs):
			cor.append(
				np.matmul(\
					np.expand_dims(
						amp_e*np.sin(np.mod(np.arange(fr)-i,fr)*2*PI/fr)+off_e,\
						axis=0
					),g
				)
			)
		cor = np.concatenate(cor,axis=0)
		return cor

	def simulate_quads(self, ipr):
		# this function simulates the impulse response
		cam = self.cam
		cor = self.cor

		# find out the non-zero part
		ipr_sum = np.sum(ipr, axis=(0,1))
		idx = np.where(ipr_sum!=0)
		ipr_s = ipr[:,:,idx[0]]
		cor_s = cor[idx[0],:]

		cor_exp = np.tile(\
				np.expand_dims(np.expand_dims(cor_s,0),0),
				[cam['dimy'],cam['dimx'],1,1]
			)
		ipr_exp = np.tile(\
				np.expand_dims(ipr_s,-1),[1,1,1,4]
			)
		meas = np.sum(cor_exp * ipr_exp, 2)
		# phase and depth
		Q = np.zeros((cam['dimy'], cam['dimx']))
		I = np.zeros((cam['dimy'], cam['dimx']))
		for i in range(len(cam['phase'])):
			Q += meas[:,:,i] * np.sin(cam['phase'][i])
			I += meas[:,:,i] * np.cos(cam['phase'][i])

		# the generalized form of phase stepping
		phase = np.arctan2(Q,I)
		ampl = 2*np.sqrt(Q**2+I**2)/len(cam['phase'])
		depth = phase * cam['T']/2/PI * C /2

		# save some data for debugging
		self.vars['ipr'] = ipr
		self.vars['cor'] = cor
		self.vars['meas'] = meas
		self.vars['phase'] = phase
		self.vars['ampl'] = ampl
		self.vars['depth']	= depth
		return

	def process(self,prop):
		self.simulate_quads(prop)
		res_dict = {
			'meas'			:		self.vars['meas'],
			'phase'			:		self.vars['phase'],
			'ampl'			:		self.vars['ampl'],
			'depth'			:		self.vars['depth'],
		}
		return res_dict

# not finished 
# TODO: New Chinese Remainder Algorithm
class cam_real_mult(cam_baseline):
	# baseline tof camera, uses square wave for emission and modulation
	# other cameras can inherit from this class
	def __init__(self,cam):
		for key in cam.keys():
			self.cam[key] = cam[key]

		# create the camera function
		self.cam['T'] 		= np.array([1.7e4, 2.5e4])
		self.cam['fr']		= np.array([\
			self.cam['T'][i]/self.cam['exp']
			for i in range(len(self.cam['T'])) 
		])
		self.cam['tabs']	= np.array([\
			self.cam['dimt']
			for i in range(len(self.cam['T']))
		])
		self.cam['amp_e']	= [4.,4.]
		self.cam['off_e']	= [2.,2.]
		self.cam['amp_m']	= [1.,1.]
		self.cam['off_m']	= [1.,1.]
		self.cam['phase'] 	= np.array([\
			[0, PI/2, PI, 3*PI/2]
			for i in range(len(self.cam['T']))
		])

		# create the camera function
		self.cor = [\
			self.cam_func(i)\
			for i in range(len(self.cam['T']))
		]

		self.vars = {}
		self.dtype= tf.float32
		# build computation graph
		self.graph = tf.Graph()
		self.session = tf.Session(graph = self.graph)
		self.build_graph()

	def cam_func(self, i):
		# precreate the camera function
		exp = self.cam['exp'] 
		# amplitude of lighting
		off_e = self.cam['off_e'][i]
		amp_e = self.cam['amp_e'][i]
		off_m = self.cam['off_m'][i]
		amp_m = self.cam['amp_m'][i]
		phase = self.cam['phase'][i]

		# constants
		tabs = self.cam['tabs'][i] # number of tabs/frs in impulse response
		fr = self.cam['fr'][i] # length of one period in frame number
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
		for i in np.arange(tabs):
			cor.append(
				np.matmul(\
					np.expand_dims(
						amp_e*np.sin(np.mod(np.arange(fr)-i,fr)*2*PI/fr)+off_e,\
						axis=0
					),g
				)
			)
		cor = np.concatenate(cor,axis=0)
		return cor

	def build_graph(self):
		# shorten the name
		cam = self.cam

		self.vars['meas_f'] = []
		self.vars['phase_f'] = []
		self.vars['ampl_f'] = []
		with self.graph.as_default():
			# inputs
			# impulse response
			ipr_init = np.zeros((cam['dimy'],cam['dimx'],cam['dimt']))
			self.ipr_in = tf.Variable(ipr_init,dtype=self.dtype)
			ipr = tf.Variable(self.ipr_in,dtype=self.dtype) 

			# camera function
			for idx in range(len(self.cam['T'])):
				cor = tf.constant(self.cor[i], dtype=self.dtype)

				# compute the raw measurement 
				cor_exp = tf.tile(\
					tf.expand_dims(tf.expand_dims(cor,0),0),
					[cam['dimy'],cam['dimx'],1,1]
				)
				ipr_exp = tf.tile(\
					tf.expand_dims(ipr,-1),[1,1,1,4]
				)
				meas = tf.reduce_sum(cor_exp * ipr_exp, 2)

				# phase and depth
				Q = tf.zeros((cam['dimy'], cam['dimx']))
				I = tf.zeros((cam['dimy'], cam['dimx']))
				for i in range(len(cam['phase'][idx])):
					Q += meas[:,:,i] * tf.sin(cam['phase'][idx].astype(np.float32))
					I += meas[:,:,i] * np.cos(cam['phase'][idx].astype(np.float32))

				# the generalized form of phase stepping
				phase = tf.atan(Q/I)
				ampl = 2*tf.sqrt(Q**2+I**2)/len(cam['phase'])

				# 
				self.vars['meas'].append(meas)
				self.vars['phase_f'].append(phase)
				self.vars['ampl_f'].append(ampl)

			# new Chinese remainder theorem
			
			depth = phase * cam['T']/2/PI * C /2

			# save some data for debugging
			self.vars['ipr'] = ipr
			self.vars['cor'] = cor
			self.vars['meas'] = meas
			self.vars['phase'] = phase
			self.vars['ampl'] = ampl
			self.vars['depth']	= depth

			# input data
			self.input_data = tf.group(\
				ipr.assign(self.ipr_in)
			)

			# 
			init_op = tf.global_variables_initializer()
			self.session.run(init_op)
		return

	def process(self,prop):
		# process data
		self.input_dict = {
			self.ipr_in		:		prop,
		}
		self.session.run(self.input_data, self.input_dict)
		res_dict = {
			'meas'			:		self.vars['meas'],
			'phase'			:		self.vars['phase'],
			'ampl'			:		self.vars['ampl'],
			'depth'			:		self.vars['depth'],
		}
		return self.session.run(res_dict)

class kinect_sin:
	# simulate kinect sensor, uses sin wave for emission and modulation
	# linear camera, uses look up table for noise
	# other cameras can inherit from this class
	def __init__(self,cam):
		self.cam = {}
		for key in cam.keys():
			self.cam[key] = cam[key]

		self.cam = kinect_sin_spec(self.cam)
		self.cor = self.cam['cor']

		self.vars = {}
		self.dtype= tf.float32

	def process(self,ipr):
		# find out the non-zero part
		ipr_sum = np.sum(ipr, axis=(0,1))
		idx = np.where(ipr_sum!=0)
		ipr_s = ipr[:,:,idx[0]]
		meas = []
		for i in range(len(self.cam['T'])):
			cor = self.cor[i]
			cor_s = cor[idx[0],:]
			cor_exp = np.tile(\
					np.expand_dims(np.expand_dims(cor_s,0),0),
					[self.cam['dimy'],self.cam['dimx'],1,1]
				)
			ipr_exp = np.tile(\
					np.expand_dims(ipr_s,-1),[1,1,1,len(self.cam['phase'][0])]
				)
			meas.append(np.sum(cor_exp * ipr_exp, 2))
		meas = np.concatenate(meas,axis=2)

		fig = plt.figure()
		for i in range(9):
			ax = fig.add_subplot(3,3,i+1)
			ax.imshow(meas[:,:,i])
		plt.show()

		# TODO: vignetting

		result = {
			'meas'	: meas
		}
		return result

class kinect_real:
	# simulate kinect sensor, uses sin wave for emission and modulation
	# linear camera, uses look up table for noise
	# other cameras can inherit from this class
	def __init__(self,cam):
		self.cam = {}
		for key in cam.keys():
			self.cam[key] = cam[key]

		self.cam = kinect_real_spec(self.cam)
		self.cor = self.cam['cor']

		self.vars = {}
		self.dtype= tf.float32

	def process(self,ipr):
		# find out the non-zero part
		ipr_sum = np.sum(ipr, axis=(0,1))
		idx = np.where(ipr_sum!=0)
		ipr_s = ipr[:,:,idx[0]]
		meas = []
		for i in range(len(self.cam['T'])):		
			cor = np.transpose(self.cor[(i*3):(i*3+3),:])
			cor_s = cor[idx[0],:]
			cor_exp = np.tile(\
					np.expand_dims(np.expand_dims(cor_s,0),0),
					[self.cam['dimy'],self.cam['dimx'],1,1]
				)
			ipr_exp = np.tile(\
					np.expand_dims(ipr_s,-1),[1,1,1,len(self.cam['phase'][0])]
				)
			meas.append(np.sum(cor_exp * ipr_exp, 2))
		meas = np.concatenate(meas,axis=2)

		result = {
			'meas'	: meas
		}
		return result

class kinect_real_tf:
	# simulate kinect sensor, uses sin wave for emission and modulation
	# linear camera, uses look up table for noise
	# other cameras can inherit from this class
	def __init__(self):
		# kinect spec
		self.cam = kinect_real_tf_spec()		

		# response graph
		self.dtype= tf.float32
		self.rg = self.res_graph()
		self.dir = '../params/kinect/'

		# delay map
		self.cam['delay'] = np.loadtxt(self.dir+'delay.txt',delimiter=',')

		# vig map
		self.cam['vig'] = np.loadtxt(self.dir+'vig.txt',delimiter=',')

		# gains
		self.cam['raw_max'] = 3500 # this brightness will be projected to 
		self.cam['map_max'] = 3500 # this brightness will be the threshold for the kinect output
		self.cam['lut_max'] = 3800 # this is the ending of lut table
		self.cam['sat'] = 32767 # this is the saturated brightness

		# noise sampling
		self.cam['noise_samp'] = np.loadtxt(self.dir+'noise_samp_2000_notail.txt',delimiter=',')
		self.cam['val_lut'] = np.arange(self.cam['noise_samp'].shape[1])-\
			(self.cam['noise_samp'].shape[1]-1)/2

		# gain and noise graph
		self.gng = self.gain_noise_graph()
		self.gg = self.gain_graph()

		# initialize the session
		self.sess = tf.Session()
		self.sess.run(tf.global_variables_initializer())
		return

	def res_graph(self):
		# shorten the name
		cam = self.cam

		# the fastest way currently to generate raw measuerment
		ipr_s = tf.placeholder(\
			self.dtype,
			[None],
			name='ipr_s'
		)

		ipr_idx = tf.placeholder(\
			tf.int64,
			[3, None],
			name='ipr_idx'
		)

		# camera function
		cha_num = 9
		cor = tf.placeholder(\
			self.dtype,
			[cha_num,None], 
			name='cor',
		)
		
		meas = []
		for i in range(cha_num):
			# expand and compute measurement
			cor_cha = cor[i,:]
			cor_exp = tf.gather(cor_cha, ipr_idx[2,:])

			# compute measurement
			tmp = cor_exp * ipr_s
			tmp = tf.SparseTensor(tf.transpose(ipr_idx), tmp, [cam['dimy'],cam['dimx'],tf.reduce_max(ipr_idx[2,:])])
			tmp1 = tf.sparse_reduce_sum(tmp,2)
			meas.append(tmp1)

		# 
		meas = tf.stack(meas, 2)

		return {'meas':meas,'ipr_s':ipr_s,'ipr_idx':ipr_idx,'cor':cor,'tmp':tmp,'tmp1':tmp1}

	def res_delay_vig_graph(self):
		# shorten the name
		cam = self.cam

		# the fastest way currently to generate raw measuerment
		ipr_s = tf.placeholder(\
			self.dtype,
			[None],
			name='ipr_s'
		)

		ipr_idx = tf.placeholder(\
			tf.int64,
			[3, None],
			name='ipr_idx'
		)

		delay_idx = tf.placeholder(\
			self.dtype,
			[None],
			name='delay_idx'
		)
		final_idx = tf.cast(ipr_idx[2,:],self.dtype)+delay_idx

		vig = tf.constant(self.cam['vig'],self.dtype)

		# camera function
		cha_num = 9
		cor = tf.placeholder(\
			self.dtype,
			[cha_num,None], 
			name='cor',
		)
		
		meas = []
		for i in range(cha_num):
			# expand and compute measurement
			# cor_cha = cor[i,:]
			# cor_exp = tf.gather(cor_cha, ipr_idx[2,:])
			cor_exp = tf.py_func(self.f[i],[final_idx], tf.float64)
			cor_exp = tf.cast(cor_exp, self.dtype)

			# compute measurement
			tmp = cor_exp * ipr_s
			tmp = tf.SparseTensor(tf.transpose(ipr_idx), tmp, [cam['dimy'],cam['dimx'],tf.reduce_max(ipr_idx[2,:])])
			tmp1 = tf.sparse_reduce_sum(tmp,2)
			meas.append(tmp1/vig)

		# 
		meas = tf.stack(meas, 2)

		return {\
			'meas':meas,
			'ipr_s':ipr_s,
			'ipr_idx':ipr_idx,
			'delay_idx':delay_idx,
			'cor':cor,
			'tmp':tmp,
			'tmp1':tmp1,
		}

	def res_delay_vig_motion_graph(self):
		# shorten the name
		cam = self.cam

		# the fastest way currently to generate raw measuerment
		ipr_s = tf.placeholder(\
			self.dtype,
			[9,None],
			name='ipr_s'
		)

		ipr_idx = tf.placeholder(\
			tf.int64,
			[3, None],
			name='ipr_idx'
		)

		delay_idx = tf.placeholder(\
			self.dtype,
			[9,None],
			name='delay_idx'
		)
		

		vig = tf.constant(self.cam['vig'],self.dtype)

		# camera function
		cha_num = 9
		cor = tf.placeholder(\
			self.dtype,
			[cha_num,None], 
			name='cor',
		)
		
		meas = []
		for i in range(cha_num):
			# expand and compute measurement
			final_idx = tf.cast(ipr_idx[2,:],self.dtype)+delay_idx[i,:]
			cor_exp = tf.py_func(self.f[i],[final_idx], tf.float64)
			cor_exp = tf.cast(cor_exp, self.dtype)

			# compute measurement
			tmp = cor_exp * ipr_s[i,:]
			tmp = tf.SparseTensor(tf.transpose(ipr_idx), tmp, [cam['dimy'],cam['dimx'],tf.reduce_max(ipr_idx[2,:])])
			tmp1 = tf.sparse_reduce_sum(tmp,2)
			meas.append(tmp1/vig)

		# 
		meas = tf.stack(meas, 2)

		return {\
			'meas':meas,
			'ipr_s':ipr_s,
			'ipr_idx':ipr_idx,
			'delay_idx':delay_idx,
			'cor':cor,
			'tmp':tmp,
			'tmp1':tmp1,
		}

	def gain_noise_graph(self):
		# shorten the name
		cam = self.cam

		# gain
		raw_max = self.cam['raw_max']
		map_max = self.cam['map_max']
		lut_max = self.cam['lut_max']

		# noise
		noise_samp = tf.constant(\
			self.cam['noise_samp'],
			tf.int32,
		)
		val_lut = tf.constant(\
			self.cam['val_lut'],
			dtype=self.dtype,
		)

		# input
		meas_i = tf.placeholder(\
			self.dtype, 
			[cam['dimy'],cam['dimx'],9],
			name='meas',
		) 

		# adjust gain
		meas = meas_i * map_max / raw_max

		# add noise
		msk = tf.less(tf.abs(meas),lut_max)
		idx = tf.where(tf.abs(meas)<lut_max) # for modifying the values
		hf = tf.cast((tf.shape(noise_samp)[1]-1)/2,self.dtype)
		mean_idx = tf.cast(tf.boolean_mask(meas,msk)+hf, tf.int32)
		samp_idx = tf.cast(tf.random_uniform(\
			tf.shape(mean_idx),minval=0,maxval=self.cam['noise_samp'].shape[0],dtype=tf.int32\
		),tf.int32)
		idx_lut = tf.stack([samp_idx,mean_idx],1)
		idx_n = tf.gather_nd(noise_samp, idx_lut)
		noise = tf.gather(val_lut, idx_n, name='noise_samp')

		# use sparse matrix to add noise
		noise_s = tf.SparseTensor(idx, noise, tf.cast(tf.shape(meas),tf.int64))
		noise_s = tf.sparse_tensor_to_dense(noise_s)
		meas = tf.cast(noise_s, tf.int32)

		# thresholding
		idx_thre = tf.where(tf.abs(meas)<map_max)
		flg = tf.ones(tf.shape(idx_thre[:,0]),tf.int32)
		flg_s = tf.SparseTensor(idx_thre, flg, tf.cast(tf.shape(meas),tf.int64))
		flg_s = tf.sparse_tensor_to_dense(flg_s)
		meas = meas * flg_s + (1-flg_s)*map_max

		# normalize to make max to equal to one
		meas_o = meas / map_max

		res_dict = {
			'meas_i'	:	meas_i, 
			'meas_o'	:	meas_o, 
			'noise'		:	noise,
			'mean_idx'	:	mean_idx,
			'idx_n'		:	idx_n,
			'idx_lut'	:	idx_lut,
			'noise_s'	:	noise_s,
		}

		return res_dict

	def gain_graph(self):
		# shorten the name
		cam = self.cam

		# gain
		raw_max = self.cam['raw_max']
		map_max = self.cam['map_max']
		lut_max = self.cam['lut_max']

		# input
		meas_i = tf.placeholder(\
			self.dtype, 
			[cam['dimy'],cam['dimx'],9],
			name='meas',
		) 

		# adjust gain
		meas = meas_i * map_max / raw_max

		# thresholding
		idx_thre = tf.where(tf.abs(meas)<map_max)
		flg = tf.ones(tf.shape(idx_thre[:,0]),tf.int32)
		flg_s = tf.SparseTensor(idx_thre, flg, tf.cast(tf.shape(meas),tf.int64))
		flg_s = tf.sparse_tensor_to_dense(flg_s)
		meas = tf.cast(meas, tf.int32) * flg_s + (1-flg_s)*map_max

		# normalize to make max to equal to one
		meas_o = meas / map_max

		res_dict = {
			'meas_i'	:	meas_i, 
			'meas_o'	:	meas_o, 
		}

		return res_dict

	def process(self,cam,ipr_idx,ipr_s,scenes,depth_true):
		# camera function: find out the non-zero part
		self.cam['dimt'] = cam['dimt']
		self.cam['exp'] = cam['exp']
		cor = compute_cor(self.cam)

		# obtain the raw measurement
		max_len = int(2e6) # max number for a GPU
		meas = np.zeros((self.cam['dimy'],self.cam['dimx'],9))
		for i in range(0,len(ipr_s),max_len):
			end = min(len(ipr_s),i+max_len)
			meas += self.sess.run(\
				self.rg['meas'],
				feed_dict={\
					self.rg['ipr_s']:ipr_s[i:end],\
					self.rg['ipr_idx']:np.array(ipr_idx)[:,i:end],\
					self.rg['cor']:cor,\
				}
			)

		# gain and noise
		meas = self.sess.run(self.gng['meas_o'],feed_dict={self.gng['meas_i']:meas})

		result = {
			'meas'	: meas
		}
		return result

	def process_no_noise(self,cam,ipr_idx,ipr_s,scenes,depth_true):
		# camera function: find out the non-zero part
		self.cam['dimt'] = cam['dimt']
		self.cam['exp'] = cam['exp']
		cor = compute_cor(self.cam)

		# obtain the raw measurement
		max_len = int(2e6) # max number for a GPU
		meas = np.zeros((self.cam['dimy'],self.cam['dimx'],9))
		for i in range(0,len(ipr_s),max_len):
			end = min(len(ipr_s),i+max_len)
			meas += self.sess.run(\
				self.rg['meas'],
				feed_dict={\
					self.rg['ipr_s']:ipr_s[i:end],\
					self.rg['ipr_idx']:np.array(ipr_idx)[:,i:end],\
					self.rg['cor']:cor,\
				}
			)

		# gain and noise
		meas = self.sess.run(self.gg['meas_o'],feed_dict={self.gg['meas_i']:meas})

		result = {
			'meas'	: meas
		}
		return result

	def process_gt(self,cam,depth_true):
		raw_max = self.cam['raw_max']

		# camera function
		self.cam['dimt'] = cam['dimt']
		self.cam['exp'] = cam['exp']
		cor = compute_cor(self.cam)

		# resize the true depth
		t = depth_true / (C/2)
		t_idx = t / self.cam['exp']
		t_idx[np.where(depth_true<1e-4)] = np.nan
		t_idx[np.where(t_idx>cor.shape[1])] = np.nan
		t_idx = scipy.misc.imresize(t_idx,(cam['dimy'],cam['dimx']),mode='F')

		# create the delay function
		self.f = []
		for i in range(cor.shape[0]):
			self.f.append(scipy.interpolate.interp1d(np.arange(cor.shape[1]),cor[i,:]))

		meas = [self.f[i](t_idx) for i in range(cor.shape[0])]
		meas = np.stack(meas, 2)

		# normalize and change the gain
		meas /= self.cam['raw_max']

		# deprecate the invalid part
		meas[np.where(np.isnan(meas))] = 0

		result = {
			'meas': meas
		}
		return result

	def process_gt_vig(self,cam,depth_true):
		raw_max = self.cam['raw_max']

		# camera function
		self.cam['dimt'] = cam['dimt']
		self.cam['exp'] = cam['exp']
		cor = compute_cor(self.cam)

		# resize the true depth
		t = depth_true / (C/2)
		t_idx = t / self.cam['exp']
		t_idx[np.where(depth_true<1e-4)] = np.nan
		t_idx[np.where(t_idx>cor.shape[1])] = np.nan
		t_idx = scipy.misc.imresize(t_idx,(cam['dimy'],cam['dimx']),mode='F')

		# create the delay function
		self.f = []
		for i in range(cor.shape[0]):
			self.f.append(scipy.interpolate.interp1d(np.arange(cor.shape[1]),cor[i,:]))

		meas = [self.f[i](t_idx)/self.cam['vig'] for i in range(cor.shape[0])]
		meas = np.stack(meas, 2)

		# # normalize based on the gain
		# meas /= np.nanmax(np.abs(meas))

		# normalize and change the gain
		meas /= self.cam['raw_max']

		# deprecate the invalid part
		meas[np.where(np.isnan(meas))] = 0

		result = {
			'meas': meas
		}
		return result

	def process_gt_vig_dist_surf(self,cam,ipr_idx,ipr_s,scenes,depth_true):
		# camera function: find out the non-zero part
		self.cam['dimt'] = cam['dimt']
		self.cam['exp'] = cam['exp']
		cor = compute_cor(self.cam)

		# find the first nonzero time frame of each pixel
		y=ipr_idx[0]
		x=ipr_idx[1]
		idx = y*self.cam['dimx']+x
		idx_u, I = np.unique(idx, return_index=True)

		ipr_idx = (ipr_idx[0][(I,)], ipr_idx[1][(I,)], ipr_idx[2][(I,)])
		ipr_s = ipr_s[(I,)]

		# obtain the raw measurement
		max_len = int(2e6) # max number for a GPU
		meas = np.zeros((self.cam['dimy'],self.cam['dimx'],9))
		for i in range(0,len(ipr_s),max_len):
			end = min(len(ipr_s),i+max_len)
			meas += self.sess.run(\
				self.rg['meas'],
				feed_dict={\
					self.rg['ipr_s']:ipr_s[i:end],\
					self.rg['ipr_idx']:np.array(ipr_idx)[:,i:end],\
					self.rg['cor']:cor,\
				}
			)
		
		# vignetting
		vig = np.tile(np.expand_dims(self.cam['vig'],-1),[1,1,9])
		meas /= vig

		# normalize and change the gain
		meas /= self.cam['raw_max']

		# deprecate the invalid part
		meas[np.where(np.isnan(meas))] = 0

		result = {
			'meas': meas
		}
		return result

	def process_gt_vig_dist_surf_mapmax(self,cam,ipr_idx,ipr_s,scenes,depth_true):
		# camera function: find out the non-zero part
		self.cam['dimt'] = cam['dimt']
		self.cam['exp'] = cam['exp']
		cor = compute_cor(self.cam)

		# find the first nonzero time frame of each pixel
		y=ipr_idx[0]
		x=ipr_idx[1]
		idx = y*self.cam['dimx']+x
		idx_u, I = np.unique(idx, return_index=True)

		ipr_idx = (ipr_idx[0][(I,)], ipr_idx[1][(I,)], ipr_idx[2][(I,)])
		ipr_s = ipr_s[(I,)]

		# obtain the raw measurement
		max_len = int(2e6) # max number for a GPU
		meas = np.zeros((self.cam['dimy'],self.cam['dimx'],9))
		for i in range(0,len(ipr_s),max_len):
			end = min(len(ipr_s),i+max_len)
			meas += self.sess.run(\
				self.rg['meas'],
				feed_dict={\
					self.rg['ipr_s']:ipr_s[i:end],\
					self.rg['ipr_idx']:np.array(ipr_idx)[:,i:end],\
					self.rg['cor']:cor,\
				}
			)
		
		# vignetting
		vig = np.tile(np.expand_dims(self.cam['vig'],-1),[1,1,9])
		meas /= vig

		# normalize and change the gain
		meas /= self.cam['raw_max']

		# deprecate the invalid part
		meas[np.where(np.isnan(meas))] = 0

		result = {
			'meas': meas
		}
		return result	

	def process_one_bounce(self,cam,ipr_idx,ipr_s,scenes,depth_true):
		# camera function: find out the non-zero part
		self.cam['dimt'] = cam['dimt']
		self.cam['exp'] = cam['exp']
		cor = compute_cor(self.cam)
		cam = nonlinear_adjust(self.cam,cor)

		# find the first nonzero time frame of each pixel
		y=ipr_idx[0]
		x=ipr_idx[1]
		idx = y*self.cam['dimx']+x
		idx_u, I = np.unique(idx, return_index=True)

		ipr_idx = (ipr_idx[0][(I,)], ipr_idx[1][(I,)], ipr_idx[2][(I,)])
		ipr_s = ipr_s[(I,)]

		# obtain the raw measurement
		max_len = int(2e6) # max number for a GPU
		meas = np.zeros((self.cam['dimy'],self.cam['dimx'],9))
		for i in range(0,len(ipr_s),max_len):
			end = min(len(ipr_s),i+max_len)
			meas += self.sess.run(\
				self.rg['meas'],
				feed_dict={\
					self.rg['ipr_s']:ipr_s[i:end],\
					self.rg['ipr_idx']:np.array(ipr_idx)[:,i:end],\
					self.rg['cor']:cor,\
				}
			)

		# gain and noise
		meas = self.sess.run(self.gg['meas_o'],feed_dict={self.gg['meas_i']:meas})

		result = {
			'meas': meas
		}
		return result

	def process_one_bounce_noise(self,cam,ipr_idx,ipr_s,scenes,depth_true):
		# camera function: find out the non-zero part
		self.cam['dimt'] = cam['dimt']
		self.cam['exp'] = cam['exp']
		cor = compute_cor(self.cam)
		cam = nonlinear_adjust(self.cam,cor)

		# find the first nonzero time frame of each pixel
		y=ipr_idx[0]
		x=ipr_idx[1]
		idx = y*self.cam['dimx']+x
		idx_u, I = np.unique(idx, return_index=True)

		ipr_idx = (ipr_idx[0][(I,)], ipr_idx[1][(I,)], ipr_idx[2][(I,)])
		ipr_s = ipr_s[(I,)]

		# obtain the raw measurement
		max_len = int(2e6) # max number for a GPU
		meas = np.zeros((self.cam['dimy'],self.cam['dimx'],9))
		for i in range(0,len(ipr_s),max_len):
			end = min(len(ipr_s),i+max_len)
			meas += self.sess.run(\
				self.rg['meas'],
				feed_dict={\
					self.rg['ipr_s']:ipr_s[i:end],\
					self.rg['ipr_idx']:np.array(ipr_idx)[:,i:end],\
					self.rg['cor']:cor,\
				}
			)

		# gain and noise
		meas = self.sess.run(self.gng['meas_o'],feed_dict={self.gng['meas_i']:meas})

		result = {
			'meas': meas
		}
		return result

	#########################################################
	# the four process functions below are what you need
	def process_delay_vig_gain_noise(self,cam,ipr_idx,ipr_s,scenes,depth_true):
		# camera function: find out the non-zero part
		self.cam['dimt'] = cam['dimt']
		self.cam['exp'] = cam['exp']
		cor = compute_cor(self.cam)

		# create the delay function
		self.f = []
		for i in range(cor.shape[0]):
			self.f.append(scipy.interpolate.interp1d(np.arange(cor.shape[1]),cor[i,:]))

		if not hasattr(self, 'rdvg'):
			self.rdvg = self.res_delay_vig_graph()

		# compute delay index and interpolates the correlation
		delay_idx = self.cam['delay'][ipr_idx[0:2]]
		delay_idx /= (C/2)
		delay_idx /= self.cam['exp']

		# obtain the raw measurement
		max_len = int(1e7) # max number for a GPU
		meas = np.zeros((self.cam['dimy'],self.cam['dimx'],9))
		for i in range(0,len(ipr_s),max_len):
			end = min(len(ipr_s),i+max_len)
			meas += self.sess.run(\
				self.rdvg['meas'],
				feed_dict={\
					self.rdvg['ipr_s']:ipr_s[i:end],\
					self.rdvg['ipr_idx']:np.array(ipr_idx)[:,i:end],\
					self.rdvg['delay_idx']:delay_idx[i:end],\
					self.rdvg['cor']:cor,\
				}
			)

		# gain and noise
		meas = self.sess.run(self.gng['meas_o'],feed_dict={self.gng['meas_i']:meas})

		result = {
			'meas'	: meas
		}
		return result

	def process_delay_vig_gain(self,cam,ipr_idx,ipr_s,scenes,depth_true):
		# camera function: find out the non-zero part
		self.cam['dimt'] = cam['dimt']
		self.cam['exp'] = cam['exp']
		cor = compute_cor(self.cam)

		# create the delay function
		self.f = []
		for i in range(cor.shape[0]):
			self.f.append(scipy.interpolate.interp1d(np.arange(cor.shape[1]),cor[i,:]))

		if not hasattr(self, 'rdvg'):
			self.rdvg = self.res_delay_vig_graph()

		# compute delay index and interpolates the correlation
		delay_idx = self.cam['delay'][ipr_idx[0:2]]
		delay_idx /= (C/2)
		delay_idx /= self.cam['exp']

		# obtain the raw measurement
		max_len = int(1e7) # max number for a GPU
		meas = np.zeros((self.cam['dimy'],self.cam['dimx'],9))
		for i in range(0,len(ipr_s),max_len):
			end = min(len(ipr_s),i+max_len)
			meas += self.sess.run(\
				self.rdvg['meas'],
				feed_dict={\
					self.rdvg['ipr_s']:ipr_s[i:end],\
					self.rdvg['ipr_idx']:np.array(ipr_idx)[:,i:end],\
					self.rdvg['delay_idx']:delay_idx[i:end],\
					self.rdvg['cor']:cor,\
				}
			)

		# gain and noise
		meas = self.sess.run(self.gg['meas_o'],feed_dict={self.gg['meas_i']:meas})

		result = {
			'meas'	: meas
		}
		return result

	def process_gt_delay_vig_gain_noise(self,cam,ipr_idx,ipr_s,scenes,depth_true):
		# camera function: find out the non-zero part
		self.cam['dimt'] = cam['dimt']
		self.cam['exp'] = cam['exp']
		cor = compute_cor(self.cam)
		cam = nonlinear_adjust(self.cam,cor)

		# find the first nonzero time frame of each pixel
		y=ipr_idx[0]
		x=ipr_idx[1]
		idx = y*self.cam['dimx']+x
		idx_u, I = np.unique(idx, return_index=True)

		ipr_idx = (ipr_idx[0][(I,)], ipr_idx[1][(I,)], ipr_idx[2][(I,)])
		ipr_s = ipr_s[(I,)]

		# create the delay function
		self.f = []
		for i in range(cor.shape[0]):
			self.f.append(scipy.interpolate.interp1d(np.arange(cor.shape[1]),cor[i,:]))

		if not hasattr(self, 'rdvg'):
			self.rdvg = self.res_delay_vig_graph()

		# compute delay index and interpolates the correlation
		delay_idx = self.cam['delay'][ipr_idx[0:2]]
		delay_idx /= (C/2)
		delay_idx /= self.cam['exp']

		# obtain the raw measurement
		max_len = int(1e7) # max number for a GPU
		meas = np.zeros((self.cam['dimy'],self.cam['dimx'],9))
		for i in range(0,len(ipr_s),max_len):
			end = min(len(ipr_s),i+max_len)
			meas += self.sess.run(\
				self.rdvg['meas'],
				feed_dict={\
					self.rdvg['ipr_s']:ipr_s[i:end],\
					self.rdvg['ipr_idx']:np.array(ipr_idx)[:,i:end],\
					self.rdvg['delay_idx']:delay_idx[i:end],\
					self.rdvg['cor']:cor,\
				}
			)

		# gain and noise
		meas = self.sess.run(self.gng['meas_o'],feed_dict={self.gng['meas_i']:meas})

		result = {
			'meas': meas
		}
		return result

	def process_gt_delay_vig_dist_surf_mapmax(self,cam,ipr_idx,ipr_s,scenes,depth_true):
		# camera function: find out the non-zero part
		self.cam['dimt'] = cam['dimt']
		self.cam['exp'] = cam['exp']
		cor = compute_cor(self.cam)
		cam = nonlinear_adjust(self.cam,cor)

		# find the first nonzero time frame of each pixel
		y=ipr_idx[0]
		x=ipr_idx[1]
		idx = y*self.cam['dimx']+x
		idx_u, I = np.unique(idx, return_index=True)

		ipr_idx = (ipr_idx[0][(I,)], ipr_idx[1][(I,)], ipr_idx[2][(I,)])
		ipr_s = ipr_s[(I,)]

		# create the delay function
		self.f = []
		for i in range(cor.shape[0]):
			self.f.append(scipy.interpolate.interp1d(np.arange(cor.shape[1]),cor[i,:]))

		if not hasattr(self, 'rdvg'):
			self.rdvg = self.res_delay_vig_graph()

		# compute delay index and interpolates the correlation
		delay_idx = self.cam['delay'][ipr_idx[0:2]]
		delay_idx /= (C/2)
		delay_idx /= self.cam['exp']

		# obtain the raw measurement
		max_len = int(1e7) # max number for a GPU
		meas = np.zeros((self.cam['dimy'],self.cam['dimx'],9))
		for i in range(0,len(ipr_s),max_len):
			end = min(len(ipr_s),i+max_len)
			meas += self.sess.run(\
				self.rdvg['meas'],
				feed_dict={\
					self.rdvg['ipr_s']:ipr_s[i:end],\
					self.rdvg['ipr_idx']:np.array(ipr_idx)[:,i:end],\
					self.rdvg['delay_idx']:delay_idx[i:end],\
					self.rdvg['cor']:cor,\
				}
			)

		# normalize and change the gain
		meas /= self.cam['raw_max']

		result = {
			'meas': meas
		}
		return result

	#########################################################
	# this is used for generating dynamic scenes
	def process_motion_delay_vig_gain(self,cam,ipr_idx,ipr_s,scenes,depth_true,motion_delay, max_t):
		# camera function: find out the non-zero part
		self.cam['dimt'] = max(cam['dimt'], int(max_t))
		self.cam['exp'] = cam['exp']
		cor = compute_cor(self.cam)

		# create the delay function
		self.f = []
		for i in range(cor.shape[0]):self.f.append(scipy.interpolate.interp1d(np.arange(cor.shape[1]),cor[i,:]))

		if not hasattr(self, 'rdvgm'):self.rdvmg = self.res_delay_vig_motion_graph()

		# compute delay index and interpolates the correlation
		delay_idx = self.cam['delay'][ipr_idx[0:2]]
		delay_idx /= (C/2)
		delay_idx /= self.cam['exp']

		# compute motion delay
		delay_indices = []
		for i in range(9):delay_indices.append(delay_idx+motion_delay[i,:])
		delay_idx = np.stack(delay_indices,0)

		# obtain the raw measurement
		max_len = int(1e7) # max number for a GPU
		meas = np.zeros((self.cam['dimy'],self.cam['dimx'],9))

		for i in range(0,ipr_s.shape[1],max_len):
			end = min(ipr_s.shape[1],i+max_len)
			meas += self.sess.run(\
				self.rdvmg['meas'],
				feed_dict={\
					self.rdvmg['ipr_s']:ipr_s[:,i:end],\
					self.rdvmg['ipr_idx']:np.array(ipr_idx)[:,i:end],\
					self.rdvmg['delay_idx']:delay_idx[:,i:end],\
					self.rdvmg['cor']:cor,\
				}
			)

		result = {
			'meas'	: meas
		}
		return result

	def add_gain_noise(self, meas):
		# gain and noise
		meas = self.sess.run(self.gng['meas_o'],feed_dict={self.gng['meas_i']:meas})
		return meas

	def dist_to_depth(self, dist):
		# this function converts the distance to camera center to depth w.r.t.
		# the camera plane
		cam = self.cam
		xx,yy = np.meshgrid(np.arange(dist.shape[1]),np.arange(dist.shape[0]))
		xc = (dist.shape[1]-1)/2
		yc = (dist.shape[0]-1)/2
		coeff = np.tan(cam['fov_x']/2/180*np.pi)
		xx = (xx - xc)/dist.shape[1]*cam['dimx']/((cam['dimx']-1)/2) * coeff
		yy = (yy - yc)/dist.shape[0]*cam['dimy']/((cam['dimx']-1)/2) * coeff
		z_multiplier = 1/np.sqrt(xx**2+yy**2+1)
		depth = dist * z_multiplier
		return depth

	def depth_to_dist(self, depth):
		# this function converts the distance to camera center to depth w.r.t.
		# the camera plane
		cam = self.cam
		xx,yy = np.meshgrid(np.arange(depth.shape[1]),np.arange(depth.shape[0]))
		xc = (depth.shape[1]-1)/2
		yc = (depth.shape[0]-1)/2
		coeff = np.tan(cam['fov_x']/2/180*np.pi)
		xx = (xx - xc)/depth.shape[1]*cam['dimx']/((cam['dimx']-1)/2) * coeff
		yy = (yy - yc)/depth.shape[0]*cam['dimy']/((cam['dimx']-1)/2) * coeff
		z_multiplier = np.sqrt(xx**2+yy**2+1)
		dist = depth * z_multiplier
		return dist

class deeptof(kinect_real_tf):
	def __init__(self):
		cam = {}
		self.dir = '../params/deeptof/'

		# camera dict
		cam['dimx'] = 512
		cam['dimy'] = 424
		cam['fov_x'] = 70

		# gains
		cam['raw_max'] = 3500 # this brightness will be projected to 
		cam['map_max'] = 3500 # this brightness will be the threshold for the kinect output
		cam['lut_max'] = 3800 # this is the ending of lut table
		cam['sat'] = 32767 # this is the saturated brightness

		# noise sampling
		cam['noise_samp'] = np.loadtxt(self.dir+'noise_samp_2000_notail.txt',delimiter=',')
		cam['val_lut'] = np.arange(cam['noise_samp'].shape[1])-\
			(cam['noise_samp'].shape[1]-1)/2

		# load the camera function
		# camera function
		cam['T'] 		= np.array([50000,]) # the period T = 1/frequency, the unit is 10^-12 sec
		cam['phase'] 	= -np.array([\
			[0*PI, 1/2*PI, PI, 3/2*PI],
		])									# the phase delay of each raw measurement
		cam['A'] = np.array([1824,])		# the amplitude of the sinusoidal, 1824 is the same as the kinect
		cam['amb']	= np.array([0,0])			# ambient light amplitude
		self.cam = cam

		# response graph
		self.dtype= tf.float32
		self.rg = self.res_graph()
		self.gng = self.gain_noise_graph()

		# initialize the session
		self.sess = tf.Session()
		self.sess.run(tf.global_variables_initializer())
		return

	def res_graph(self):
		# shorten the name
		cam = self.cam

		# the fastest way currently to generate raw measuerment
		ipr_s = tf.placeholder(\
			self.dtype,
			[None],
			name='ipr_s'
		)

		ipr_idx = tf.placeholder(\
			tf.int64,
			[3, None],
			name='ipr_idx'
		)

		# camera function
		cha_num = 4
		cor = tf.placeholder(\
			self.dtype,
			[cha_num,None], 
			name='cor',
		)
		
		meas = []
		for i in range(cha_num):
			# expand and compute measurement
			cor_cha = cor[i,:]
			cor_exp = tf.gather(cor_cha, ipr_idx[2,:])

			# compute measurement
			tmp = cor_exp * ipr_s
			tmp = tf.SparseTensor(tf.transpose(ipr_idx), tmp, [cam['dimy'],cam['dimx'],tf.reduce_max(ipr_idx[2,:])])
			tmp1 = tf.sparse_reduce_sum(tmp,2)
			meas.append(tmp1)

		# 
		meas = tf.stack(meas, 2)

		return {'meas':meas,'ipr_s':ipr_s,'ipr_idx':ipr_idx,'cor':cor,'tmp':tmp,'tmp1':tmp1}

	def gain_noise_graph(self):
		# shorten the name
		cam = self.cam

		# gain
		raw_max = self.cam['raw_max']
		map_max = self.cam['map_max']
		lut_max = self.cam['lut_max']

		# noise
		noise_samp = tf.constant(\
			self.cam['noise_samp'],
			tf.int32,
		)
		val_lut = tf.constant(\
			self.cam['val_lut'],
			dtype=self.dtype,
		)

		# input
		meas_i = tf.placeholder(\
			self.dtype, 
			[cam['dimy'],cam['dimx'],4],
			name='meas',
		) 

		# adjust gain
		meas = meas_i * map_max / raw_max

		# add noise
		msk = tf.less(tf.abs(meas),lut_max)
		idx = tf.where(tf.abs(meas)<lut_max) # for modifying the values
		hf = tf.cast((tf.shape(noise_samp)[1]-1)/2,self.dtype)
		mean_idx = tf.cast(tf.boolean_mask(meas,msk)+hf, tf.int32)
		samp_idx = tf.cast(tf.random_uniform(\
			tf.shape(mean_idx),minval=0,maxval=self.cam['noise_samp'].shape[0],dtype=tf.int32\
		),tf.int32)
		idx_lut = tf.stack([samp_idx,mean_idx],1)
		idx_n = tf.gather_nd(noise_samp, idx_lut)
		noise = tf.gather(val_lut, idx_n, name='noise_samp')

		# use sparse matrix to add noise
		noise_s = tf.SparseTensor(idx, noise, tf.cast(tf.shape(meas),tf.int64))
		noise_s = tf.sparse_tensor_to_dense(noise_s)
		meas = tf.cast(noise_s, tf.int32)

		# thresholding
		idx_thre = tf.where(tf.abs(meas)<map_max)
		flg = tf.ones(tf.shape(idx_thre[:,0]),tf.int32)
		flg_s = tf.SparseTensor(idx_thre, flg, tf.cast(tf.shape(meas),tf.int64))
		flg_s = tf.sparse_tensor_to_dense(flg_s)
		meas = meas * flg_s + (1-flg_s)*map_max

		# normalize to make max to equal to one
		meas_o = meas / map_max

		res_dict = {
			'meas_i'	:	meas_i, 
			'meas_o'	:	meas_o, 
			'noise'		:	noise,
			'mean_idx'	:	mean_idx,
			'idx_n'		:	idx_n,
			'idx_lut'	:	idx_lut,
			'noise_s'	:	noise_s,
		}

		return res_dict

	def compute_cor_deeptof(self, cam):
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
		return np.array(cor)

	def process_gain_noise(self,cam,ipr_idx,ipr_s,scenes,depth_true):
		# camera function: find out the non-zero part
		self.cam['dimt'] = cam['dimt']
		self.cam['exp'] = cam['exp']
		cor = self.compute_cor_deeptof(self.cam)

		# create the delay function
		self.f = []
		for i in range(cor.shape[0]):
			self.f.append(scipy.interpolate.interp1d(np.arange(cor.shape[1]),cor[i,:]))

		if not hasattr(self, 'rg'):
			self.rg = self.res_graph()

		# obtain the raw measurement
		max_len = int(1e7) # max number for a GPU
		meas = np.zeros((self.cam['dimy'],self.cam['dimx'],4))
		for i in range(0,len(ipr_s),max_len):
			end = min(len(ipr_s),i+max_len)
			meas += self.sess.run(\
				self.rg['meas'],
				feed_dict={\
					self.rg['ipr_s']:ipr_s[i:end],\
					self.rg['ipr_idx']:np.array(ipr_idx)[:,i:end],\
					self.rg['cor']:cor,\
				}
			)

		# gain and noise
		meas = self.sess.run(self.gng['meas_o'],feed_dict={self.gng['meas_i']:meas})

		result = {
			'meas'	: meas
		}
		return result

	def process_gt_gain_noise(self,cam,ipr_idx,ipr_s,scenes,depth_true):
		# camera function: find out the non-zero part
		self.cam['dimt'] = cam['dimt']
		self.cam['exp'] = cam['exp']
		cor = self.compute_cor_deeptof(self.cam)

		# find the first nonzero time frame of each pixel
		y=ipr_idx[0]
		x=ipr_idx[1]
		idx = y*self.cam['dimx']+x
		idx_u, I = np.unique(idx, return_index=True)

		ipr_idx = (ipr_idx[0][(I,)], ipr_idx[1][(I,)], ipr_idx[2][(I,)])
		ipr_s = ipr_s[(I,)]

		# create the delay function
		self.f = []
		for i in range(cor.shape[0]):
			self.f.append(scipy.interpolate.interp1d(np.arange(cor.shape[1]),cor[i,:]))

		if not hasattr(self, 'rg'):
			self.rg = self.res_graph()

		# obtain the raw measurement
		max_len = int(1e7) # max number for a GPU
		meas = np.zeros((self.cam['dimy'],self.cam['dimx'],4))
		for i in range(0,len(ipr_s),max_len):
			end = min(len(ipr_s),i+max_len)
			meas += self.sess.run(\
				self.rg['meas'],
				feed_dict={\
					self.rg['ipr_s']:ipr_s[i:end],\
					self.rg['ipr_idx']:np.array(ipr_idx)[:,i:end],\
					self.rg['cor']:cor,\
				}
			)

		result = {
			'meas'	: meas
		}
		return result

class phasor(kinect_real_tf):
	def __init__(self):
		cam = {}
		self.dir = '../params/phasor/'

		# camera dict
		cam['dimx'] = 512
		cam['dimy'] = 424
		cam['fov_x'] = 70

		# gains
		cam['raw_max'] = 3500 # this brightness will be projected to 
		cam['map_max'] = 3500 # this brightness will be the threshold for the kinect output
		cam['lut_max'] = 3800 # this is the ending of lut table
		cam['sat'] = 32767 # this is the saturated brightness

		# noise sampling
		cam['noise_samp'] = np.loadtxt(self.dir+'noise_samp_2000_notail.txt',delimiter=',')
		cam['val_lut'] = np.arange(cam['noise_samp'].shape[1])-\
			(cam['noise_samp'].shape[1]-1)/2

		# load the camera function
		# camera function
		cam['T'] 		= np.array([940.7, 967.1]) # the period T = 1/frequency, the unit is 10^-12 sec
		cam['phase'] 	= -np.array([\
			[0*PI, 1/2*PI, PI, 3/2*PI],
			[0*PI, 1/2*PI, PI, 3/2*PI]
		])										# the phase delay of each raw measurement
		cam['A'] = np.array([1824,1824])		# the amplitude of the sinusoidal, 1824 is the same as the kinect
		cam['amb']	= np.array([0,0])			# ambient light amplitude
		self.cam = cam

		# response graph
		self.dtype= tf.float32
		self.rg = self.res_graph()
		self.gng = self.gain_noise_graph()
		self.gg = self.gain_graph()

		# initialize the session
		self.sess = tf.Session()
		self.sess.run(tf.global_variables_initializer())
		return

	def res_graph(self):
		# shorten the name
		cam = self.cam

		# the fastest way currently to generate raw measuerment
		ipr_s = tf.placeholder(\
			self.dtype,
			[None],
			name='ipr_s'
		)

		ipr_idx = tf.placeholder(\
			tf.int64,
			[3, None],
			name='ipr_idx'
		)

		# camera function
		cha_num = 8
		cor = tf.placeholder(\
			self.dtype,
			[cha_num,None], 
			name='cor',
		)
		
		meas = []
		for i in range(cha_num):
			# expand and compute measurement
			cor_cha = cor[i,:]
			cor_exp = tf.gather(cor_cha, ipr_idx[2,:])

			# compute measurement
			tmp = cor_exp * ipr_s
			tmp = tf.SparseTensor(tf.transpose(ipr_idx), tmp, [cam['dimy'],cam['dimx'],tf.reduce_max(ipr_idx[2,:])])
			tmp1 = tf.sparse_reduce_sum(tmp,2)
			meas.append(tmp1)

		# 
		meas = tf.stack(meas, 2)

		return {'meas':meas,'ipr_s':ipr_s,'ipr_idx':ipr_idx,'cor':cor,'tmp':tmp,'tmp1':tmp1}
	
	def gain_noise_graph(self):
		# shorten the name
		cam = self.cam

		# gain
		raw_max = self.cam['raw_max']
		map_max = self.cam['map_max']
		lut_max = self.cam['lut_max']

		# noise
		noise_samp = tf.constant(\
			self.cam['noise_samp'],
			tf.int32,
		)
		val_lut = tf.constant(\
			self.cam['val_lut'],
			dtype=self.dtype,
		)

		# input
		meas_i = tf.placeholder(\
			self.dtype, 
			[cam['dimy'],cam['dimx'],8],
			name='meas',
		) 

		# adjust gain
		meas = meas_i * map_max / raw_max

		# add noise
		msk = tf.less(tf.abs(meas),lut_max)
		idx = tf.where(tf.abs(meas)<lut_max) # for modifying the values
		hf = tf.cast((tf.shape(noise_samp)[1]-1)/2,self.dtype)
		mean_idx = tf.cast(tf.boolean_mask(meas,msk)+hf, tf.int32)
		samp_idx = tf.cast(tf.random_uniform(\
			tf.shape(mean_idx),minval=0,maxval=self.cam['noise_samp'].shape[0],dtype=tf.int32\
		),tf.int32)
		idx_lut = tf.stack([samp_idx,mean_idx],1)
		idx_n = tf.gather_nd(noise_samp, idx_lut)
		noise = tf.gather(val_lut, idx_n, name='noise_samp')

		# use sparse matrix to add noise
		noise_s = tf.SparseTensor(idx, noise, tf.cast(tf.shape(meas),tf.int64))
		noise_s = tf.sparse_tensor_to_dense(noise_s)
		meas = tf.cast(noise_s, tf.int32)

		# thresholding
		idx_thre = tf.where(tf.abs(meas)<map_max)
		flg = tf.ones(tf.shape(idx_thre[:,0]),tf.int32)
		flg_s = tf.SparseTensor(idx_thre, flg, tf.cast(tf.shape(meas),tf.int64))
		flg_s = tf.sparse_tensor_to_dense(flg_s)
		meas = meas * flg_s + (1-flg_s)*map_max

		# normalize to make max to equal to one
		meas_o = meas / map_max

		res_dict = {
			'meas_i'	:	meas_i, 
			'meas_o'	:	meas_o, 
			'noise'		:	noise,
			'mean_idx'	:	mean_idx,
			'idx_n'		:	idx_n,
			'idx_lut'	:	idx_lut,
			'noise_s'	:	noise_s,
		}

		return res_dict

	def gain_graph(self):
		# shorten the name
		cam = self.cam

		# gain
		raw_max = self.cam['raw_max']
		map_max = self.cam['map_max']
		lut_max = self.cam['lut_max']

		# input
		meas_i = tf.placeholder(\
			self.dtype, 
			[cam['dimy'],cam['dimx'],8],
			name='meas',
		) 

		# adjust gain
		meas = meas_i * map_max / raw_max

		# thresholding
		idx_thre = tf.where(tf.abs(meas)<map_max)
		flg = tf.ones(tf.shape(idx_thre[:,0]),tf.int32)
		flg_s = tf.SparseTensor(idx_thre, flg, tf.cast(tf.shape(meas),tf.int64))
		flg_s = tf.sparse_tensor_to_dense(flg_s)
		meas = tf.cast(meas, tf.int32) * flg_s + (1-flg_s)*map_max

		# normalize to make max to equal to one
		meas_o = meas / map_max

		res_dict = {
			'meas_i'	:	meas_i, 
			'meas_o'	:	meas_o, 
		}

		return res_dict

	def compute_cor_deeptof(self, cam):
		# time frame
		cam['dimt'] += 20
		cam['tabs']	= np.array([\
			cam['dimt']
			for i in range(len(cam['T']))
		])
		cam['t'] = (np.arange(cam['dimt']))*cam['exp']

		# create the camera function
		cor = [\
			cam['A'][i]*np.sin(2*PI/cam['T'][i]*cam['t']-cam['phase'][i,j])+cam['amb'][i] \
			for i in range(len(cam['T'])) for j in range(len(cam['phase'][i,:]))\
		]
		return np.array(cor)

	def process_gain_noise(self,cam,ipr_idx,ipr_s,scenes,depth_true):
		# this function generates raw measurement of phasor WITH noise, reflection
		# camera function: find out the non-zero part
		self.cam['dimt'] = cam['dimt']
		self.cam['exp'] = cam['exp']
		cor = self.compute_cor_deeptof(self.cam)

		if not hasattr(self, 'rg'):
			self.rg = self.res_graph()

		# obtain the raw measurement
		max_len = int(1e7) # max number for a GPU
		meas = np.zeros((self.cam['dimy'],self.cam['dimx'],8))
		for i in range(0,len(ipr_s),max_len):
			end = min(len(ipr_s),i+max_len)
			meas += self.sess.run(\
				self.rg['meas'],
				feed_dict={\
					self.rg['ipr_s']:ipr_s[i:end],\
					self.rg['ipr_idx']:np.array(ipr_idx)[:,i:end],\
					self.rg['cor']:cor,\
				}
			)

		# gain and noise
		meas = self.sess.run(self.gng['meas_o'],feed_dict={self.gng['meas_i']:meas})

		result = {
			'meas'	: meas
		}
		return result

	def process_gain(self,cam,ipr_idx,ipr_s,scenes,depth_true):
		# this function generates raw measurement of phasor with only reflection
		# camera function: find out the non-zero part
		self.cam['dimt'] = cam['dimt']
		self.cam['exp'] = cam['exp']
		cor = self.compute_cor_deeptof(self.cam)

		if not hasattr(self, 'rg'):
			self.rg = self.res_graph()

		# obtain the raw measurement
		max_len = int(1e7) # max number for a GPU
		meas = np.zeros((self.cam['dimy'],self.cam['dimx'],8))
		for i in range(0,len(ipr_s),max_len):
			end = min(len(ipr_s),i+max_len)
			meas += self.sess.run(\
				self.rg['meas'],
				feed_dict={\
					self.rg['ipr_s']:ipr_s[i:end],\
					self.rg['ipr_idx']:np.array(ipr_idx)[:,i:end],\
					self.rg['cor']:cor,\
				}
			)

		# gain and noise
		meas = self.sess.run(self.gg['meas_o'],feed_dict={self.gg['meas_i']:meas})

		result = {
			'meas'	: meas
		}
		return result

	def process_gt_gain(self,cam,ipr_idx,ipr_s,scenes,depth_true):
		# this function generates raw measurement of phasor with no noise nor reflection
		# camera function: find out the non-zero part
		self.cam['dimt'] = cam['dimt']
		self.cam['exp'] = cam['exp']
		cor = self.compute_cor_deeptof(self.cam)

		# find the first nonzero time frame of each pixel
		y=ipr_idx[0]
		x=ipr_idx[1]
		idx = y*self.cam['dimx']+x
		idx_u, I = np.unique(idx, return_index=True)

		ipr_idx = (ipr_idx[0][(I,)], ipr_idx[1][(I,)], ipr_idx[2][(I,)])
		ipr_s = ipr_s[(I,)]

		if not hasattr(self, 'rg'):
			self.rg = self.res_graph()

		# obtain the raw measurement
		max_len = int(1e7) # max number for a GPU
		meas = np.zeros((self.cam['dimy'],self.cam['dimx'],8))
		for i in range(0,len(ipr_s),max_len):
			end = min(len(ipr_s),i+max_len)
			meas += self.sess.run(\
				self.rg['meas'],
				feed_dict={\
					self.rg['ipr_s']:ipr_s[i:end],\
					self.rg['ipr_idx']:np.array(ipr_idx)[:,i:end],\
					self.rg['cor']:cor,\
				}
			)

		# gain and noise
		meas = self.sess.run(self.gg['meas_o'],feed_dict={self.gg['meas_i']:meas})

		result = {
			'meas'	: meas
		}
		return result

	def process_gt_gain_noise(self,cam,ipr_idx,ipr_s,scenes,depth_true):
		# this function generates raw measurement of phasor with only noise
		# camera function: find out the non-zero part
		self.cam['dimt'] = cam['dimt']
		self.cam['exp'] = cam['exp']
		cor = self.compute_cor_deeptof(self.cam)

		# find the first nonzero time frame of each pixel
		y=ipr_idx[0]
		x=ipr_idx[1]
		idx = y*self.cam['dimx']+x
		idx_u, I = np.unique(idx, return_index=True)

		ipr_idx = (ipr_idx[0][(I,)], ipr_idx[1][(I,)], ipr_idx[2][(I,)])
		ipr_s = ipr_s[(I,)]

		if not hasattr(self, 'rg'):
			self.rg = self.res_graph()

		# obtain the raw measurement
		max_len = int(1e7) # max number for a GPU
		meas = np.zeros((self.cam['dimy'],self.cam['dimx'],8))
		for i in range(0,len(ipr_s),max_len):
			end = min(len(ipr_s),i+max_len)
			meas += self.sess.run(\
				self.rg['meas'],
				feed_dict={\
					self.rg['ipr_s']:ipr_s[i:end],\
					self.rg['ipr_idx']:np.array(ipr_idx)[:,i:end],\
					self.rg['cor']:cor,\
				}
			)

		# gain and noise
		meas = self.sess.run(self.gng['meas_o'],feed_dict={self.gng['meas_i']:meas})

		result = {
			'meas'	: meas
		}
		return result
