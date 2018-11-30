# license:  Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
#           Licensed under the CC BY-NC-SA 4.0 license
#           (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode). 

# utility functions
import numpy as np
import cv2
import pdb
import os, json, glob
import pickle

C = 0.000299792458
PI = 3.14159265358
# from tof_class import *
"""
This function transform the single-point view depthmap
to a 3D mesh data
"""
def map2mesh(depthmap,cam):
	xx,yy = np.meshgrid(np.arange(cam['dimx']),np.arange(cam['dimy']))
	xx = (xx/cam['dimx'] - 0.5) * cam['size'][1]/cam['vp']
	yy = (yy/cam['dimy'] - 0.5) * cam['size'][0]/cam['vp']
	rr = np.sqrt(xx**2 + yy**2)
	XX = xx / np.sqrt(1 + rr**2) * depthmap
	YY = yy / np.sqrt(1 + rr**2) * depthmap
	ZZ = 1 / np.sqrt(1 + rr**2) * depthmap
	return XX,YY,ZZ

"""
This function generates the command to run the transient rendering
"""
def gen_cmd(program,cam,scene,filename):
	command = []
	command.append(program)
	command.append(cam['mode'])
	command.append(' '.join(['-film-size-x', tof_str(cam['dimx'])]))
	command.append(' '.join(['-film-size-y', tof_str(cam['dimy'])]))
	command.append(' '.join(['-film-size-t', tof_str(cam['dimt'])]))
	command.append(' '.join(['-film-exposure', tof_str(cam['exp'])]))
	command.append(' '.join(['-film-offset', tof_str(cam['toffset'])]))
	command.append(' '.join([tof_str(cam['single_pixel'][i]) for i in range(len(cam['single_pixel']))]))
	command.append(' '.join([tof_str(cam['tsamp'][i]) for i in range(len(cam['tsamp']))]))
	command.append(' '.join(['-camera-spp', tof_str(cam['spp'])]))
	command.append(' '.join(['-bidirectional-path-tracing', tof_str(cam['vptsamp'])]))
	command.append(' '.join(['-camera-position', tof_str(cam['pos'])]))
	command.append(' '.join(['-camera-focus', tof_str(cam['lookat'])]))
	command.append(' '.join(['-camera-view-plane', tof_str(cam['vp'])]))
	command.append(' '.join(['-camera-up', tof_str(cam['up'])]))
	command.append(' '.join([tof_str(cam['bouncing'][i]) for i in range(len(cam['bouncing']))]))


	command.append(' '.join([tof_str(scene['light'][i]) for i in range(len(scene['light']))]))
	command.append(' '.join(['-scattering-level', tof_str(scene['scatlevel'])]))
	command.append(' '.join(['-homogeneous-medium', tof_str(scene['media'])]))
	command.append(' '.join(['-hg', tof_str(scene['hg'])]))

	command.append(' '.join([\
		tof_str(scene['setup'][i][j])
		for i in range(len(scene['setup']))
		for j in range(len(scene['setup'][i]))
	]))
	command.append(' '.join(['-film-name', tof_str(filename)]))
	command.append(' '.join(['-new-seed', tof_str(int(np.random.uniform()*100))]))
	return ' '.join(command)

def tof_str(a):
	if (type(a) is list) or (type(a) is np.ndarray):
		# if a is an array
		return ' '.join([str(a[i]) for i in range(len(a))])
	else:
		return str(a)

"""
Visualize the light propagation
"""
def vis_prop(prop):
	t_range = prop.shape[2]
	t = 0
	cv2.namedWindow("Light propagation", cv2.WINDOW_NORMAL)
	while(t < t_range):
		tmp = cv2.cvtColor(prop[:,:,t].astype(np.float32)*30,cv2.COLOR_GRAY2BGR)
		t_s = 0.8
		t_h = int(20*t_s)
		tmp = cv2.putText(\
			tmp, \
			str(t), \
			(0,t_h+3), \
			cv2.FONT_HERSHEY_DUPLEX , \
			t_s, \
			(1,1,1)\
		)
		cv2.imshow("Light propagation",tmp)
		c=cv2.waitKey(20) % 0x110000
		if chr(c).lower() == 'q':
			break
		t += 1
	return

"""
Find the non-zero time frame of the impulse response to speed up
"""
def find_nonzero(prop):
	light_amount = np.sum(prop,(0,1))
	total_amount = np.sum(light_amount)
	acc_amount = 0
	st = 0
	st_flag = 0
	ed = 0
	ed_thre = 0.999
	for i in range(light_amount.shape[0]):
		# judge whether the light has arrived at the scene
		if st_flag == 0 and light_amount[i] != 0:
			st = i
			st_flag = 1
		
		# judge whether the light has reached ed_thre of the total light
		acc_amount += light_amount[i]
		if acc_amount/total_amount > ed_thre:
			ed = i
			break
	ed += 1
	return prop_data[:,:,st:ed],st,ed

"""
Conduct manual correlation from start index to end index
"""	
def manual_corr(a, b, st, ed):
	if not (st>0 and ed<a.shape[0] and a.shape[0]==b.shape[0]):
		raise ValueError('Invalid Size!')
	c = np.zeros(ed-st)
	n = a.shape[0]
	idx = 0
	for i in np.arange(st,ed):
		c[idx] = sum(a[0:n-i]*b[i:n])+sum(a[n-i:n]*b[0:i])
		idx += 1
	return c


def tile_images(im, rows, cols, norm=True):
	if len(im.shape)==4:
		im = np.reshape(im, (im.shape[0], im.shape[1], -1))
	# total images
	rows_total = (im.shape[0]+1)*rows
	cols_total = (im.shape[1]+1)*cols
	img = np.zeros((rows_total, cols_total))*np.nan
	for i in range(rows):
		for j in range(cols):
			y = i * (im.shape[0]+1)
			x = j * (im.shape[1]+1)
			if (i*cols+j) < im.shape[2]:
				if norm:
					# normalize
					tmp = im[:,:,i*cols+j]
					tmp -= np.nanmin(tmp)
					tmp /= np.nanmax(tmp)
					img[y:y+im.shape[0],x:x+im.shape[1]]=tmp
				else:
					img[y:y+im.shape[0],x:x+im.shape[1]]=im[:,:,i*cols+j]
	return img


