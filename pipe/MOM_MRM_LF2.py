# license:  Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
#           Licensed under the CC BY-NC-SA 4.0 license
#           (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode). 

# this code simulates the time-of-flight data
# all time unit are picoseconds (1 picosec = 1e-12 sec)
import sys
sys.path.insert(0,'../sim/')
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

PI = 3.14159265358979323846
raw_depth_new = 0
flg = False

dtype = tf.float32

def select_objects():
    check = False
    data_dir = '../FLAT/kinect/'
    array_dir = '../FLAT/trans_render/static/'

    # background is selected from far corners
    f = open('../FLAT/kinect/list/motion_background.txt','r')
    message = f.read()
    files = message.split('\n')
    trains = files[0:-1]
    back_scenes = [data_dir+train for train in trains]

    # foreground is selected from objects
    f = open('../FLAT/kinect/list/motion_foreground.txt','r')
    message = f.read()
    files = message.split('\n')
    trains = files[0:-1]
    fore_scenes = [data_dir+train for train in trains]
    
    rand_num = np.random.uniform()
    while(check==False):
        back_idx = np.random.choice(len(back_scenes),1,replace=False)
        fore_num = np.random.choice(1,1,replace=False)[0]+1
        fore_idx = np.random.choice(len(fore_scenes),fore_num,replace=False)

        # put the scenes together, with a probability to not using the background
        if rand_num < 0.3:
            scenes = [back_scenes[idx] for idx in back_idx]+[fore_scenes[idx] for idx in fore_idx]

            # check the distance between them are larger than 0.1m
            depths = []
            msks = []
            for scene in scenes:
                with open(scene[0:-16]+'gt/'+scene[-16::],'rb') as f:
                    gt=np.fromfile(f, dtype=np.float32)
                depths.append(np.reshape(gt,(424*4,512*4)))
                msks.append((depths[-1]==0)*99999)

            diff = [depths[i]-depths[j]+msks[i]+msks[j] for i in range(len(msks)) for j in range(i+1,len(msks))]
            diff = np.stack(diff,0)
            if diff.min()>0.1 and diff[0:fore_num].min()>0:check = True
        elif rand_num > 0.3 and rand_num < 0.6:
            scenes = [fore_scenes[idx] for idx in fore_idx]
            check = True
        elif rand_num > 0.6:
            scenes = [back_scenes[idx] for idx in back_idx]
            check = True
    
    # # show the scene
    # depths = np.stack(depths,0)
    # depths[np.where(depths==0)] = 999999
    # idx = np.argmin(depths,0)
    # y = np.arange(depths[0].shape[0])
    # x = np.arange(depths[0].shape[1])
    # xx, yy = np.meshgrid(x,y)
    # pts = [idx.flatten(), yy.flatten(), xx.flatten()]
    # depth = np.reshape(depths[pts], xx.shape)
    # depth[np.where(depth>10)] = 0
    # plt.figure();plt.imshow(depth);plt.show()

    return scenes

def select_objects_val():
    check = False
    data_dir = '../FLAT/kinect/'
    array_dir = '../FLAT/trans_render/static/'

    # background is selected from far corners
    f = open('../FLAT/kinect/list/motion_background.txt','r')
    message = f.read()
    files = message.split('\n')
    trains = files[0:-1]
    back_scenes = [data_dir+train for train in trains]

    # foreground is selected from objects
    f = open('../FLAT/kinect/list/val.txt','r')
    message = f.read()
    files = message.split('\n')
    trains = files[0:-1]
    fore_scenes = [data_dir+train for train in trains]

    rand_num = np.random.uniform()
    while(check==False):
        back_idx = np.random.choice(len(back_scenes),1,replace=False)
        fore_num = np.random.choice(1,1,replace=False)[0]+1
        fore_idx = np.random.choice(len(fore_scenes),fore_num,replace=False)

        # put the scenes together, with a probability to not using the background
        if rand_num < 0.3:
            scenes = [back_scenes[idx] for idx in back_idx]+[fore_scenes[idx] for idx in fore_idx]

            # check the distance between them are larger than 0.1m
            depths = []
            msks = []
            for scene in scenes:
                with open(scene[0:-16]+'gt/'+scene[-16::],'rb') as f:
                    gt=np.fromfile(f, dtype=np.float32)
                depths.append(np.reshape(gt,(424*4,512*4)))
                msks.append((depths[-1]==0)*99999)

            diff = [depths[i]-depths[j]+msks[i]+msks[j] for i in range(len(msks)) for j in range(i+1,len(msks))]
            diff = np.stack(diff,0)
            if diff.min()>0.1 and diff[0:fore_num].min()>0:check = True
        elif rand_num > 0.3 and rand_num < 0.6:
            scenes = [fore_scenes[idx] for idx in fore_idx]
            check = True
        elif rand_num > 0.6:
            scenes = [back_scenes[idx] for idx in back_idx]
            check = True
    
    # # show the scene
    # depths = np.stack(depths,0)
    # depths[np.where(depths==0)] = 999999
    # idx = np.argmin(depths,0)
    # y = np.arange(depths[0].shape[0])
    # x = np.arange(depths[0].shape[1])
    # xx, yy = np.meshgrid(x,y)
    # pts = [idx.flatten(), yy.flatten(), xx.flatten()]
    # depth = np.reshape(depths[pts], xx.shape)
    # depth[np.where(depth>10)] = 0
    # plt.figure();plt.imshow(depth);plt.show()

    return scenes

def select_objects_rand(scenes,nums=1):
    num_total = len(scenes)
    idx = np.random.choice(num_total, nums, replace=False).astype(np.int32)
    return [scenes[i] for i in idx]

def data_augment_th(scene_ns, array_dir, tof_cam, text_flg = False):
    global flg
    # v_flg is used to judge whether the velocity is out of bound
    v_flg = False
    while (v_flg == False):
        v_flg = True
        # first loading each scene, and we will combine them then
        meass = []
        meass_gt = []
        depths = []
        msks = []
        vs = []
        for scene_n in scene_ns:
            print('Augmenting scene', scene_n)
            ## load all data
            # if the raw file does not exist, just find one and use
            if not os.path.exists(array_dir+scene_n[-16:]+'.pickle'):
                scenes = glob.glob(array_dir+'*.pickle')
                with open(scenes[0],'rb') as f:
                    data = pickle.load(f)
                cam = data['cam']

                # separately read the true depth and true rendering
                with open(scene_n[0:-16]+'gt/'+scene_n[-16::],'rb') as f:
                    gt=np.fromfile(f, dtype=np.float32)
                depth_true = np.reshape(gt,(cam['dimy']*4,cam['dimx']*4))

                with open(scene_n[0:-16]+'ideal/'+scene_n[-16::],'rb') as f:
                    meas_gt=np.fromfile(f, dtype=np.int32)
                meas_gt = np.reshape(meas_gt,(cam['dimy'],cam['dimx'],9)).astype(np.float32)
            else:
                with open(array_dir+scene_n[-16::]+'.pickle','rb') as f:
                    data = pickle.load(f)
                program = data['program']
                cam = data['cam']
                cam_t = data['cam_t']
                scene = data['scene']
                depth_true = data['depth_true']
                prop_idx = data['prop_idx']
                prop_s = data['prop_s'] 
                res_gt = tof_cam.process_gt_delay_vig_dist_surf_mapmax(cam, prop_idx, prop_s, scene, depth_true)
                meas_gt = res_gt['meas']

            # directly read pregenerate raw measurement
            with open(scene_n[0:-16]+'full/'+scene_n[-16::],'rb') as f:
                meas=np.fromfile(f, dtype=np.int32)
            meas = np.reshape(meas,(cam['dimy'],cam['dimx'],9)).astype(np.float32)
            msk = kinect_mask().astype(np.float32)
            meas = [meas[:,:,i]*msk for i in range(meas.shape[2])]
            meas = np.stack(meas,-1)
            meas = meas / tof_cam.cam['map_max']

            meas_gt = [meas_gt[:,:,i]*msk for i in range(meas_gt.shape[2])]
            meas_gt = np.stack(meas_gt,-1)
            meas_gt = meas_gt / tof_cam.cam['map_max']
            # meas = meas[::-1,:,:]

            # reduce the resolution of the depth
            depth_true_s = scipy.misc.imresize(\
                depth_true,\
                meas.shape[0:2],\
                mode='F'\
            )
            depth_true_s = tof_cam.dist_to_depth(depth_true_s)

            # load the mask and classification
            with open(scene_n[0:-16]+'msk'+'/'+scene_n[-16:],'rb') as f:
                msk_array=np.fromfile(f, dtype=np.float32)
            msk_array = np.reshape(msk_array,(cam['dimy'],cam['dimx'],4))
            msk = {}
            msk['background'] = msk_array[:,:,0]
            msk['edge'] = msk_array[:,:,1]
            msk['noise'] = msk_array[:,:,2]
            msk['reflection'] = msk_array[:,:,3]

            # compute mask
            # msk_true_s = msk['background'] * msk['edge']
            msk_true = (depth_true > 1e-4).astype(np.float32)
            msk_true_s = scipy.misc.imresize(\
                msk_true,\
                meas.shape[0:2],\
                mode='F'\
            )
            msk_true_s = (msk_true_s > 0.999).astype(np.float32)
            depth_true_s = msk_true_s * depth_true_s

            true = np.stack([depth_true_s,msk_true_s],2)
            true = np.concatenate([true, meas_gt], 2)

            msk = msk_true_s

            if text_flg == True:
                # add textures (simply multiply a ratio)
                texts = glob.glob('../FLAT/kinect/list/textures-curet/'+'*.png')
                idx = np.random.choice(len(texts),1,replace=False)[0]
                im_text = cv2.imread(texts[idx],0).astype(np.float32)
                im_text /= 255.
                lo = np.random.uniform(0,1) # random range
                hi = np.random.uniform(lo,1)
                im_text = im_text * (hi-lo) + lo
                im_text = scipy.misc.imresize(im_text,meas.shape[0:2],mode='F')
                im_text = np.expand_dims(im_text,-1)

                # apply the texture
                meas = meas * im_text
                meas_gt = meas_gt * im_text


            # randomly generating an in image velocity
            v = np.random.rand(2)
            detv = np.random.uniform()*5 # the biggest velocity is 2 pixels per channel
            v = v / np.sqrt(np.sum(v**2)) * detv

            # randomly generating the 6 affine transform parameters
            max_pix = 10
            mov = 10
            if np.random.uniform() > 0.3:
                while (np.abs(mov).max() >= max_pix):
                    th1 = np.random.normal(0.0,0.01,[2,2])
                    th1[0,0]+=1
                    th1[1,1]+=1
                    th2 = np.random.normal(0.0,.5,[2,1])
                    th3 = np.array([[0,0,1]])
                    th = np.concatenate([th1,th2],1)
                    th = np.concatenate([th,th3],0)
                    x = np.arange(meas.shape[1])
                    y = np.arange(meas.shape[0])
                    xx, yy = np.meshgrid(x,y)
                    mov = np.sqrt(\
                        ((1-th[0,0])*yy-th[0,1]*xx-th[0,2])**2 + \
                        ((1-th[1,1])*xx-th[1,0]*yy-th[1,2])**2
                    )*msk
            else:
                th = [[1,0,0],[0,1,0],[0,0,1]]

            # append the data
            meass.append(meas)
            meass_gt.append(meas_gt)
            depths.append(depth_true_s)
            msks.append(msk)
            vs.append(th)

        # cut out foreground that is too close to the background
        back_idx = 0
        fore_idx = np.arange(1,len(scene_ns))
        dist_min = 0.5
        for idx in fore_idx:
            diff = depths[back_idx]-depths[idx]+(1-msks[back_idx])*99
            msk_close = (diff > dist_min).astype(np.float32)
            meass[idx] = [meass[idx][:,:,i]*msk_close for i in range(9)]
            meass[idx] = np.stack(meass[idx], -1)
            meass_gt[idx] = [meass_gt[idx][:,:,i]*msk_close for i in range(9)]
            meass_gt[idx] = np.stack(meass_gt[idx], -1)
            depths[idx] = depths[idx]*msk_close
            msks[idx] = msks[idx]*msk_close

        # move the object and combine them by channel
        y = np.arange(meass[0].shape[0])
        x = np.arange(meass[0].shape[1])
        xx, yy = np.meshgrid(x,y)
        meass_new = []
        meass_old = []
        meass_gt_new = []
        vys_new = []
        vxs_new = []
        vys_inv = []
        vxs_inv = []
        msks_new = []
        msks_old = []
        depths_new = []

        mid = 4
        for i in range(9):
            meas_v = []
            meas_old_v = []
            meas_gt_v = []
            depth_v = []
            msk_v = []
            msk_old_v = []
            depth_old_v = []
            vy_v = []
            vx_v = []
            vy_inv = []
            vx_inv = []
            for j in range(len(meass)):
                # constant transformation
                th = vs[j]
                th = LA.matrix_power(th, i-mid)
                pts_y = th[0,0]*yy+th[0,1]*xx+th[0,2]
                pts_x = th[1,0]*yy+th[1,1]*xx+th[1,2]
                pts = np.stack([pts_y.flatten(), pts_x.flatten()],-1)

                f1 = scipy.interpolate.RegularGridInterpolator((y,x),meass[j][:,:,i],bounds_error=False, fill_value=0)
                meas_v.append(np.reshape(f1(pts), xx.shape))
                meas_old_v.append(meass[j][:,:,i])
                meas_gt_v.append(meass_gt[j][:,:,i])

                f2 = scipy.interpolate.RegularGridInterpolator((y,x),depths[j],bounds_error=False, fill_value=0)
                depth_v.append(np.reshape(f2(pts), xx.shape))
                depth_old_v.append(depths[j])

                f3 = scipy.interpolate.RegularGridInterpolator((y,x),msks[j],bounds_error=False, fill_value=0)
                msk_v.append(np.reshape(f3(pts), xx.shape))
                msk_old_v.append(msks[j])

                vy_v.append(pts_y - yy)
                vx_v.append(pts_x - xx)

                # generate velocity between adjacent frame
                th = vs[j]
                th = LA.matrix_power(th, i-mid)
                pts_y = th[0,0]*yy+th[0,1]*xx+th[0,2]
                pts_x = th[1,0]*yy+th[1,1]*xx+th[1,2]
                vy_inv.append(pts_y - yy)
                vx_inv.append(pts_x - xx)

                # mask out those regions that interpolates with the background
                msk_v[-1][np.where(msk_v[-1]<0.999)] = 0
                msk_old_v[-1][np.where(msk_old_v[-1]<0.999)] = 0

                # meas_v[-1] *= msk_v[-1]
                # depth_v[-1] *= msk_v[-1]
                # vy_v[-1] *= msk_v[-1]
                # vx_v[-1] *= msk_v[-1]

                # the velocity must be cropped using the ground truth mask
                vy_inv[-1] = vy_inv[-1] * msks[j].astype(np.float32)
                vx_inv[-1] = vx_inv[-1] * msks[j].astype(np.float32)

            # combine the raw measurement based on depth
            msk_v = np.stack(msk_v, -1)
            msk_old_v = np.stack(msk_old_v, -1)
            meas_v = np.stack(meas_v, -1)
            meas_old_v = np.stack(meas_old_v, -1)
            meas_gt_v = np.stack(meas_gt_v, -1)
            depth_v = np.stack(depth_v, -1)
            depth_old_v = np.stack(depth_old_v, -1)
            vy_v  = np.stack(vy_v, -1)
            vx_v  = np.stack(vx_v, -1)
            vy_inv = np.stack(vy_inv, -1)
            vx_inv = np.stack(vx_inv, -1)

            # combine 
            depth_v[np.where(depth_v == 0)] = 999999999
            idx = np.argmin(depth_v, -1)
            pts = [yy.flatten(), xx.flatten(), idx.flatten()]
            meas_new = np.reshape(meas_v[pts], xx.shape)
            vy_new = np.reshape(vy_v[pts], xx.shape)
            vx_new = np.reshape(vx_v[pts], xx.shape)
            msk_new = np.reshape(msk_v[pts], xx.shape)
            depth_new = np.reshape(depth_v[pts], xx.shape)

            depth_old_v[np.where(depth_old_v == 0)] = 999999999
            idx = np.argmin(depth_old_v, -1)
            pts = [yy.flatten(), xx.flatten(), idx.flatten()]
            vy_inv = np.reshape(vy_inv[pts], xx.shape)
            vx_inv = np.reshape(vx_inv[pts], xx.shape)
            meas_old = np.reshape(meas_old_v[pts], xx.shape)
            meas_gt_new = np.reshape(meas_gt_v[pts], xx.shape)
            msk_old = np.reshape(msk_old_v[pts], xx.shape)

            meass_new.append(meas_new)
            vys_new.append(vy_new)
            vxs_new.append(vx_new)
            msks_new.append(msk_new)
            msks_old.append(msk_old)
            depths_new.append(depth_new)
            vys_inv.append(vy_inv)
            vxs_inv.append(vx_inv)
            meass_old.append(meas_old)
            meass_gt_new.append(meas_gt_new)

            exp = 1
            if (np.abs(vy_inv).max() < (max_pix*np.abs(i-mid)+exp)) and (np.abs(vx_inv).max() < (max_pix*np.abs(i-mid)+exp)):
                v_flg = True
            else:
                v_flg = False
                break

    meas_all = np.stack(meass_new, -1)
    meas_all = meas_all[20:-20,:,:]
    meas_old_all = np.stack(meass_old, -1)
    meas_old_all = meas_old_all[20:-20,:,:]
    meass_gt_new = np.stack(meass_gt_new, -1)
    meass_gt_new = meass_gt_new[20:-20,:,:]
    vys = np.stack(vys_inv, -1)
    vxs = np.stack(vxs_inv, -1)
    vys = vys[20:-20,:,:]
    vxs = vxs[20:-20,:,:]
    vys = -vys
    vxs = -vxs

    k_shape = [\
        (\
            (max_pix*np.abs(i-mid)+exp)*2+1,
            (max_pix*np.abs(i-mid)+exp)*2+1
        ) 
        for i in range(9)
    ]

    # interpolation to make continuous kernel
    kers_y = []
    kers_x = []
    kers_yf = []
    kers_xf = []
    for k_idx in range(9):
        vy = vys[:,:,k_idx]
        vx = vxs[:,:,k_idx]
        ker_y = np.zeros(vy.shape+(k_shape[k_idx][0],1))
        ker_x = np.zeros(vx.shape+(1,k_shape[k_idx][1]))
        vy_fl = np.floor(vy)+(max_pix*np.abs(k_idx-mid)+exp)
        vy_cl = np.ceil(vy)+(max_pix*np.abs(k_idx-mid)+exp)
        vx_fl = np.floor(vx)+(max_pix*np.abs(k_idx-mid)+exp)
        vx_cl = np.ceil(vx)+(max_pix*np.abs(k_idx-mid)+exp)
        vy = vy + (max_pix*np.abs(k_idx-mid)+exp)
        vx = vx + (max_pix*np.abs(k_idx-mid)+exp)
        w_yfl = (1 - np.abs(vy - vy_fl))
        w_ycl = (1 - np.abs(vy - vy_cl))
        w_xfl = (1 - np.abs(vx - vx_fl))
        w_xcl = (1 - np.abs(vx - vx_cl))
        y = np.arange(ker_y.shape[0])
        x = np.arange(ker_x.shape[1])
        xx, yy = np.meshgrid(x,y)
        xx = xx.flatten()
        yy = yy.flatten()
        oo = np.zeros(xx.shape).astype(np.int32)
        vy_fl = np.reshape(vy_fl,(-1,1)).astype(np.int32)
        vx_fl = np.reshape(vx_fl,(-1,1)).astype(np.int32)
        vy_cl = np.reshape(vy_cl,(-1,1)).astype(np.int32)
        vx_cl = np.reshape(vx_cl,(-1,1)).astype(np.int32)
        idx0 = [yy, xx, vy_fl[:,0], oo]
        ker_y[idx0] = w_yfl.flatten()
        idx1 = [yy, xx, vy_cl[:,0], oo]
        ker_y[idx1] = w_ycl.flatten()
        idx2 = [yy, xx, oo, vx_fl[:,0]]
        ker_x[idx2] = w_xfl.flatten()
        idx3 = [yy, xx, oo, vx_cl[:,0]]
        ker_x[idx3] = w_xcl.flatten()

        ker_yf = np.reshape(ker_y, vy.shape+(-1,))
        ker_xf = np.reshape(ker_x, vx.shape+(-1,))

        kers_y.append(ker_y)
        kers_x.append(ker_x)
        kers_yf.append(ker_yf)
        kers_xf.append(ker_xf)

    kers_yf = np.concatenate(kers_yf, -1)
    kers_xf = np.concatenate(kers_xf, -1)
    meas = meas_all

    depths = np.stack(depths_new, -1)
    depths[np.where(depths>10)] = 0
    depths = depths[20:-20,:,:]
    msks = np.stack(msks_old, -1)
    msks = msks[20:-20,:,:]

    depth_true_s = depth_true_s[20:-20,:]
    true = np.concatenate([depths[:,:,mid:mid+1], meas_old_all, vys, vxs, meass_gt_new, msks[:,:,mid:mid+1]],-1)
    # true = np.concatenate([kers_yf, kers_xf, true],-1)
    ker = np.concatenate([kers_yf, kers_xf], -1)
    # normalize
    ker = ker / np.expand_dims(np.sum(ker, -1),-1) * 18

    # fig = plt.figure()
    # for i in range(9):ax=fig.add_subplot(3,3,i+1);plt.imshow(meas_gt[:,:,i])
    # fig = plt.figure()
    # for i in range(9):ax=fig.add_subplot(3,3,i+1);plt.imshow(meas_all[:,:,i])
    # plt.show()

    # # verify if the code is correct
    # cha_idx = 0
    # ims_warped = []

    # for cha_idx in range(9):
    #     meas_tmp = meas_all[:,:,cha_idx]
    #     # y convolution
    #     im_tmp = []
    #     ky_rep = np.zeros((k_shape[cha_idx][0],k_shape[cha_idx][0]))
    #     for idx in range(k_shape[cha_idx][0]):ky_rep[idx,idx] = 1
    #     ky_rep = np.reshape(ky_rep, [k_shape[cha_idx][0], 1, k_shape[cha_idx][0]])
    #     for idx in range(k_shape[cha_idx][0]):im_tmp.append(signal.convolve2d(meas_tmp, ky_rep[:,:,idx], mode='same'))
    #     im_tmp = np.stack(im_tmp,-1)
    #     ker_tmp = np.reshape(kers_y[cha_idx],kers_y[cha_idx].shape[0:2]+(k_shape[cha_idx][0],))
    #     meas_tmp = np.sum(im_tmp * ker_tmp,-1)

    #     # x convolution
    #     im_tmp = []
    #     kx_rep = np.zeros((k_shape[cha_idx][1],k_shape[cha_idx][1]))
    #     for idx in range(k_shape[cha_idx][1]):kx_rep[idx,idx] = 1
    #     kx_rep = np.reshape(kx_rep, [1, k_shape[cha_idx][1], k_shape[cha_idx][1]])
    #     for idx in range(k_shape[cha_idx][1]):im_tmp.append(signal.convolve2d(meas_tmp, kx_rep[:,:,idx], mode='same'))
    #     im_tmp = np.stack(im_tmp,-1)
    #     ker_tmp = np.reshape(kers_x[cha_idx],kers_x[cha_idx].shape[0:2]+(k_shape[cha_idx][0],))
    #     meas_tmp = np.sum(im_tmp * ker_tmp,-1)

    #     ims_warped.append(meas_tmp)

    # fig = plt.figure()
    # for i in range(9):ax=fig.add_subplot(3,3,i+1);plt.imshow(np.abs(ims_warped[i]))
    # fig = plt.figure()
    # for i in range(9):ax=fig.add_subplot(3,3,i+1);plt.imshow(np.abs(meas_old_all[:,:,i]))
    # fig = plt.figure()
    # for i in range(9):ax=fig.add_subplot(3,3,i+1);plt.imshow(np.abs(meas_all[:,:,i]))
    # fig = plt.figure()
    # for i in range(9):ax=fig.add_subplot(3,3,i+1);plt.imshow(vys[:,:,i])
    # fig = plt.figure()
    # for i in range(9):ax=fig.add_subplot(3,3,i+1);plt.imshow(vxs[:,:,i])
    # fig = plt.figure()
    # for i in range(9):ax=fig.add_subplot(3,3,i+1);plt.imshow(depths[:,:,i])
    # fig = plt.figure()
    # for i in range(9):ax=fig.add_subplot(3,3,i+1);plt.imshow(meas_old_all[:,:,i])
    # fig = plt.figure()
    # for i in range(9):ax=fig.add_subplot(3,3,i+1);plt.imshow(meass_gt_new[:,:,i])
    # fig = plt.figure()
    # for i in range(9):ax=fig.add_subplot(3,3,i+1);plt.imshow(msks[:,:,i])
    # plt.show()

    # select a part of the image
    py = int(meas.shape[0]/2)
    px = int(meas.shape[1]/2)
    meas_p = []
    true_p = []
    ker_p = []
    for iy in range(0,meas.shape[0]-py+1,int(py/2)):
        for ix in range(0,meas.shape[1]-px+1,int(px/2)):
            meas_p.append(meas[iy:(iy+py),ix:(ix+px)])
            true_p.append(true[iy:(iy+py),ix:(ix+px)])
            ker_p.append(ker[iy:(iy+py),ix:(ix+px)])
    meas_p = np.stack(meas_p, 0)
    true_p = np.stack(true_p, 0)
    ker_p = np.stack(ker_p,0)

    # # visualize the velocity
    # idx = 0
    # fig = plt.figure()
    # for i in range(9):ax = fig.add_subplot(3,3,i+1);plt.imshow(meas_p[idx,:,:,i])
    # plt.show()
    # pdb.set_trace()

    # the input of the network
    return meas, true, depths, vys, vxs

def data_augment_3d(scene_ns, array_dir, tof_cam, text_flg = False):
    global flg
    # first loading each scene, and we will combine them then
    meass = []
    depths = []
    msks = []
    vs = []
    v_flg = False
    while (v_flg == False):
        v_flg = True
        # first loading each scene, and we will combine them then
        meass = []
        depths = []
        msks = []
        vs = []
        Ps = []
        for scene_n in scene_ns:
            print('Augmenting scene', scene_n)
            ## load all data
            # if the raw file does not exist, just find one and use
            if not os.path.exists(array_dir+scene_n[-16:]+'.pickle'):
                scenes = glob.glob(array_dir+'*.pickle')
                with open(scenes[0],'rb') as f:
                    data = pickle.load(f)
                cam = data['cam']

                # separately read the true depth and true rendering
                with open(scene_n[0:-16]+'gt/'+scene_n[-16::],'rb') as f:
                    gt=np.fromfile(f, dtype=np.float32)
                depth_true = np.reshape(gt,(cam['dimy']*4,cam['dimx']*4))

                with open(scene_n[0:-16]+'ideal/'+scene_n[-16::],'rb') as f:
                    meas_gt=np.fromfile(f, dtype=np.int32)
                meas_gt = np.reshape(meas_gt,(cam['dimy'],cam['dimx'],9)).astype(np.float32)
            else:
                with open(array_dir+scene_n[-16::]+'.pickle','rb') as f:
                    data = pickle.load(f)
                program = data['program']
                cam = data['cam']
                cam_t = data['cam_t']
                scene = data['scene']
                depth_true = data['depth_true']
                prop_idx = data['prop_idx']
                prop_s = data['prop_s'] 
                res_gt = tof_cam.process_gt_delay_vig_dist_surf_mapmax(cam, prop_idx, prop_s, scene, depth_true)
                meas_gt = res_gt['meas']

            # directly read pregenerate raw measurement
            with open(scene_n[0:-16]+'full/'+scene_n[-16::],'rb') as f:
                meas=np.fromfile(f, dtype=np.int32)
            meas = np.reshape(meas,(cam['dimy'],cam['dimx'],9)).astype(np.float32)
            msk = kinect_mask().astype(np.float32)
            meas = [meas[:,:,i]*msk for i in range(meas.shape[2])]
            meas = np.stack(meas,-1)
            meas = meas / tof_cam.cam['map_max']
            # meas = meas[::-1,:,:]

            meas_gt = [meas_gt[:,:,i]*msk for i in range(meas_gt.shape[2])]
            meas_gt = np.stack(meas_gt,-1)
            meas_gt = meas_gt / tof_cam.cam['map_max']

            # reduce the resolution of the depth
            depth_true[np.where(depth_true==0)] = np.nan # deal with the mix problem at edge
            depth_true_s = scipy.misc.imresize(\
                depth_true,\
                meas.shape[0:2],\
                mode='F'\
            )
            depth_true_s = tof_cam.dist_to_depth(depth_true_s)
            depth_true_s[np.where(np.isnan(depth_true_s))] = 0

            # load the mask and classification
            with open(scene_n[0:-16]+'msk'+'/'+scene_n[-16:],'rb') as f:
                msk_array=np.fromfile(f, dtype=np.float32)
            msk_array = np.reshape(msk_array,(cam['dimy'],cam['dimx'],4))
            msk = {}
            msk['background'] = msk_array[:,:,0]
            msk['edge'] = msk_array[:,:,1]
            msk['noise'] = msk_array[:,:,2]
            msk['reflection'] = msk_array[:,:,3]

            # compute mask
            msk_true_s = msk['background'] * msk['edge']

            true = np.stack([depth_true_s,msk_true_s],2)
            true = np.concatenate([true, meas_gt], 2)

            msk = msk_true_s

            if text_flg == True:
                # add textures (simply multiply a ratio)
                texts = glob.glob('../FLAT/kinect/list/textures-curet/'+'*.png')
                idx = np.random.choice(len(texts),1,replace=False)[0]
                im_text = cv2.imread(texts[idx],0).astype(np.float32)
                im_text /= 255.
                lo = np.random.uniform(0,1) # random range
                hi = np.random.uniform(lo,1)
                im_text = im_text * (hi-lo) + lo
                im_text = scipy.misc.imresize(im_text,meas.shape[0:2],mode='F')
                im_text = np.expand_dims(im_text,-1)

                # apply the texture
                meas = meas * im_text
                meas_gt = meas_gt * im_text

            # compute the camera matrix
            xx,yy = np.meshgrid(np.arange(depth_true_s.shape[1]), np.arange(depth_true_s.shape[0]))
            ratio = depth_true_s.shape[1]
            fov = 0.7
            xx = (xx.flatten() - (xx.shape[1]-1)/2)/ratio
            yy = (yy.flatten() - (yy.shape[0]-1)/2)/ratio
            xx = xx * fov 
            yy = yy * fov

            depth_f = depth_true_s.flatten()
            idx = np.where(depth_f != 0)
            xx = xx[idx]
            yy = yy[idx]
            depth_f = depth_f[idx]
            idx = np.random.choice(len(depth_f),2000,replace=False)
            xx = xx[idx]
            yy = yy[idx]
            depth_f = depth_f[idx]

            pts_3d = np.stack([xx*depth_f, yy*depth_f, depth_f, np.ones(depth_f.shape)],-1)
            pts_2d = np.stack([xx, yy, np.ones(depth_f.shape)],-1)
            # use the DLT algorithm
            a00 = np.zeros(pts_3d.shape)
            a01 = -pts_2d[:,2:3]*pts_3d
            a02 = pts_2d[:,1:2]*pts_3d
            a10 = -a01
            a11 = np.zeros(pts_3d.shape)
            a12 = -pts_2d[:,0:1]*pts_3d
            a20 = -a02
            a21 = -a12
            a22 = np.zeros(pts_3d.shape)
            a0 = np.concatenate([a00, a01, a02],1)
            a1 = np.concatenate([a10, a11, a12],1)
            a2 = np.concatenate([a20, a21, a22],1)
            A = np.concatenate([a0, a1, a2], 0)
            U,s,vh=np.linalg.svd(A, full_matrices =False)
            v = vh.T
            P = np.reshape(v[:,-1],[3,4])
            pts_2d_reproj = np.matmul(pts_3d,P.T)
            pts_2d_reproj /= pts_2d_reproj[:,-1::]
            reproj_err = np.sum(np.abs(pts_2d_reproj - pts_2d))
            print('Reprojection error:',reproj_err)

            # # visualize the 3d points
            # from mpl_toolkits.mplot3d import axes3d
            # fig = plt.figure()
            # ax = fig.add_subplot(121, projection='3d')
            # ax.scatter(pts_3d[:,0], pts_3d[:,1], pts_3d[:,2],'.')
            # ax.set_aspect('equal')

            # ax = fig.add_subplot(122)
            # plt.imshow(meas[:,:,0])
            # plt.show()

            # randomly generating the 6 affine transform parameters
            max_pix = 5
            max_mov_m = 0.03
            mov = 10
            while (np.abs(mov).max() >= max_mov_m):
                th1 = np.random.normal(0.0,0.01,[3,3])
                th1[0,0]+=1
                th1[1,1]+=1
                th1[2,2]+=1
                th2 = np.random.normal(0.0,.01,[3,1])
                th3 = np.array([[0,0,0,1]])
                th = np.concatenate([th1,th2],1)
                th = np.concatenate([th,th3],0)
                
                Y = pts_3d[:,0]
                X = pts_3d[:,1]
                Z = pts_3d[:,2]
                pts_3d_new = np.matmul(pts_3d, th.T)
                mov = np.sqrt(np.sum((pts_3d_new - pts_3d)**2,1))
                # print(th)
                # mov = np.sqrt(\
                #     ((1-th[0,0])*Y-th[0,1]*X-th[0,2]*Z-th[0,3])**2+
                #     (-th[1,0]*Y+(1-th[1,1])*X-th[1,2]*Z-th[1,3])**2+
                #     (-th[2,0]*Y-th[2,1]*X+(1-th[2,2])*Z-th[2,3])**2
                # )

            # append the data
            meass.append(meas)
            depths.append(depth_true_s)
            msks.append(msk)
            vs.append(th)
            Ps.append(P)

        # move the object and combine them by channel
        y = np.arange(meass[0].shape[0])
        x = np.arange(meass[0].shape[1])
        xx, yy = np.meshgrid(x,y)
        meass_new = []
        meass_old = []
        vys_new = []
        vxs_new = []
        vys_inv = []
        vxs_inv = []
        msks_new = []
        depths_new = []

        mid = 4
        for i in range(9):
            meas_v = []
            meas_old_v = []
            depth_v = []
            msk_v = []
            depth_old_v = []
            vy_v = []
            vx_v = []
            vy_inv = []
            vx_inv = []
            for j in range(len(meass)):
                # constant transformation
                # notice that the velocity is inversed here
                th = vs[j]
                th = LA.matrix_power(th, i-mid)

                # 
                xx_p = (xx - (xx.shape[1]-1)/2)/ratio
                yy_p = (yy - (yy.shape[0]-1)/2)/ratio
                zz_p = depths[j]
                xx_p = xx_p * fov * zz_p
                yy_p = yy_p * fov * zz_p
                
                xx_p = xx_p.flatten()
                yy_p = yy_p.flatten()
                zz_p = zz_p.flatten()
                idx = np.where(zz_p != 0)
                yy_p = yy_p[idx]
                xx_p = xx_p[idx]
                zz_p = zz_p[idx]

                # prepare teh data
                meas_f = meass[j][:,:,i].flatten()
                meas_f = meas_f[idx]
                depth_f = depths[j].flatten()
                depth_f = depth_f[idx]
                msk_f = msks[j].flatten()
                msk_f = msk_f[idx]

                # do the transformation
                pts_3d = np.stack([yy_p, xx_p, zz_p, np.ones(xx_p.shape)],-1)
                pts_2d_raw = np.stack([(yy.flatten())[idx], (xx.flatten())[idx]],-1)
                pts_2d = np.stack([yy_p / zz_p, xx_p / zz_p],-1)
                pts_3d_new = np.matmul(pts_3d, th.T)
                P = Ps[j]
                pts_2d_new = np.matmul(pts_3d_new,P.T)
                pts_2d_new = pts_2d_new[:,0:2]/pts_2d_new[:,2:3]
                y_p = pts_2d_new[:,0] / fov * ratio + (xx.shape[0]-1)/2
                x_p = pts_2d_new[:,1] / fov * ratio + (xx.shape[1]-1)/2
                pts_2d_new_raw = np.stack([y_p, x_p],-1)
                pts = np.stack([yy.flatten(), xx.flatten()],-1)

                # # # visualize the 3d points
                # from mpl_toolkits.mplot3d import axes3d
                # idx = np.random.choice(len(zz_p),1000,replace=False).astype(np.int32)
                # fig = plt.figure()
                # ax = fig.add_subplot(111, projection='3d')
                # ax.scatter(pts_3d[idx,0], pts_3d[idx,1], pts_3d[idx,2],c=zz_p[idx])
                # ax.scatter(pts_3d_new[idx,0], pts_3d_new[idx,1], pts_3d_new[idx,2],c=-zz_p[idx])
                # ax.set_aspect('equal')
                # plt.show()

                # cut off the regions outside 
                idx = np.where((y_p<(yy.shape[0]-1))*(y_p>0)*(x_p<(xx.shape[1]-1))*(x_p>0))
                y_pc = y_p[idx]
                x_pc = x_p[idx]

                # add a map of zerso
                zero_map = np.zeros(xx.shape)
                zero_map[(np.floor(y_pc).astype(np.int32),np.floor(x_pc).astype(np.int32))] = 1
                zero_map[(np.ceil(y_pc).astype(np.int32),np.floor(x_pc).astype(np.int32))] = 1
                zero_map[(np.floor(y_pc).astype(np.int32),np.ceil(x_pc).astype(np.int32))] = 1
                zero_map[(np.ceil(y_pc).astype(np.int32),np.ceil(x_pc).astype(np.int32))] = 1

                # plt.figure();plt.imshow(zero_map)
                # plt.figure();plt.imshow(depths[j]);plt.show()

                y_zero = yy[np.where(zero_map==0)]
                x_zero = xx[np.where(zero_map==0)]
                val_nan = np.nan*x_zero

                pts_2d_zero = np.stack([y_zero, x_zero],-1)
                pts_2d_new_full = np.concatenate([pts_2d_new_raw, pts_2d_zero],0)
                meas_f = np.concatenate([meas_f, val_nan],0)
                depth_f = np.concatenate([depth_f, val_nan],0)
                msk_f = np.concatenate([msk_f, val_nan],0)

                f1 = scipy.interpolate.griddata(pts_2d_new_full,meas_f,pts)
                meas_v.append(np.reshape(f1, xx.shape))
                meas_old_v.append(meass[j][:,:,i])

                f2 = scipy.interpolate.griddata(pts_2d_new_full,depth_f,pts)
                depth_v.append(np.reshape(f2, xx.shape))
                depth_old_v.append(depths[j])

                f3 = scipy.interpolate.griddata(pts_2d_new_full,msk_f,pts)
                msk_v.append(np.reshape(f3, xx.shape))

                # add the velocity
                vy_v.append(np.zeros(yy.shape))
                vy_v[-1][(pts_2d_raw[:,0],pts_2d_raw[:,1])] = pts_2d_new_raw[:,0] - pts_2d_raw[:,0]
                vx_v.append(np.ones(xx.shape))
                vx_v[-1][(pts_2d_raw[:,0],pts_2d_raw[:,1])] = pts_2d_new_raw[:,1] - pts_2d_raw[:,1]

                # mask out those regions that interpolates with the background
                msk_v[-1][np.where(msk_v[-1]<0.999)] = 0

                # meas_v[-1] *= msk_v[-1]
                # depth_v[-1] *= msk_v[-1]
                # vy_v[-1] *= msk_v[-1]
                # vx_v[-1] *= msk_v[-1]

            # combine the raw measurement based on depth
            msk_v = np.stack(msk_v, -1)
            meas_v = np.stack(meas_v, -1)
            meas_old_v = np.stack(meas_old_v, -1)
            depth_v = np.stack(depth_v, -1)
            depth_old_v = np.stack(depth_old_v, -1)
            vy_v  = np.stack(vy_v, -1)
            vx_v  = np.stack(vx_v, -1)

            # combine 
            depth_v[np.where(np.isnan(depth_v))] = 999999999
            idx = np.argmin(depth_v, -1)
            pts = [yy.flatten(), xx.flatten(), idx.flatten()]
            meas_new = np.reshape(meas_v[pts], xx.shape)
            vy_new = np.reshape(vy_v[pts], xx.shape)
            vx_new = np.reshape(vx_v[pts], xx.shape)
            msk_new = np.reshape(msk_v[pts], xx.shape)
            depth_new = np.reshape(depth_v[pts], xx.shape)

            # remove the 
            msk_new[np.where(np.isnan(msk_new))] = 0
            meas_new[np.where(np.isnan(meas_new))] = 0

            depth_old_v[np.where(depth_old_v == 0)] = 999999999
            idx = np.nanargmin(depth_old_v, -1)
            pts = [yy.flatten(), xx.flatten(), idx.flatten()]
            vy_inv = np.reshape(vy_v[pts], xx.shape)
            vx_inv = np.reshape(vx_v[pts], xx.shape)
            meas_old = np.reshape(meas_old_v[pts], xx.shape)

            meass_new.append(meas_new)
            vys_new.append(vy_new)
            vxs_new.append(vx_new)
            msks_new.append(msk_new)
            depths_new.append(depth_new)
            vys_inv.append(vy_inv)
            vxs_inv.append(vx_inv)
            meass_old.append(meas_old)

            # exp = 1
            # if (np.abs(vy_inv).max() < (max_pix*np.abs(i-mid)+exp)) and (np.abs(vx_inv).max() < (max_pix*np.abs(i-mid)+exp)):
            #     v_flg = True
            # else:
            #     v_flg = False
            #     break

    meas_all = np.stack(meass_new, -1)
    meas_all = meas_all[20:-20,:,:]
    meas_old_all = np.stack(meass_old, -1)
    meas_old_all = meas_old_all[20:-20,:,:]
    meas_gt = meas_gt[20:-20,:,:]
    vys = np.stack(vys_inv, -1)
    vxs = np.stack(vxs_inv, -1)
    vys = -vys
    vxs = -vxs
    vys = vys[20:-20,:,:]
    vxs = vxs[20:-20,:,:]

    meas = meas_all
    true = meas_old_all
    depth_true_s = depth_true_s[20:-20,:]
    true = np.concatenate([np.expand_dims(depth_true_s,-1),true, vys, vxs, meas_gt],-1)
    

    # # verify the velocity is correct
    # y = np.arange(vys.shape[0])
    # x = np.arange(vxs.shape[1])
    # xx, yy = np.meshgrid(x, y)
    # meas_warped = []
    # for i in range(9):
    #     y_new = yy + vys[:,:,i]
    #     x_new = xx + vxs[:,:,i]
    #     pts_new = np.stack([y_new, x_new], -1)
    #     f1 = scipy.interpolate.RegularGridInterpolator((y,x),meas_all[:,:,i],bounds_error=False, fill_value=0)
    #     meas_warped.append(np.reshape(f1(pts_new),meas_all.shape[0:2]))
    # meas_warped = np.stack(meas_warped, -1)

    # fig = plt.figure()
    # for i in range(9):ax=fig.add_subplot(3,3,i+1);plt.imshow(np.abs(meas_warped[:,:,i]))
    # fig = plt.figure()
    # for i in range(9):ax=fig.add_subplot(3,3,i+1);plt.imshow(np.abs(meas_old_all[:,:,i]))
    # fig = plt.figure()
    # for i in range(9):ax=fig.add_subplot(3,3,i+1);plt.imshow(np.abs(meas_all[:,:,i]))
    # fig = plt.figure()
    # for i in range(9):ax=fig.add_subplot(3,3,i+1);plt.imshow(vys[:,:,i])
    # fig = plt.figure()
    # for i in range(9):ax=fig.add_subplot(3,3,i+1);plt.imshow(vxs[:,:,i])
    # plt.show()

    # select a part of the image
    py = int(meas.shape[0]/2)
    px = int(meas.shape[1]/2)
    meas_p = []
    true_p = []
    for iy in range(0,meas.shape[0]-py+1,int(py/2)):
        for ix in range(0,meas.shape[1]-px+1,int(px/2)):
            meas_p.append(meas[iy:(iy+py),ix:(ix+px)])
            true_p.append(true[iy:(iy+py),ix:(ix+px)])
    meas_p = np.stack(meas_p, 0)
    true_p = np.stack(true_p, 0)

    meas = np.expand_dims(meas,0)
    true = np.expand_dims(true,0)

    # # visualize the velocity
    # idx = 0
    # fig = plt.figure()
    # for i in range(9):ax = fig.add_subplot(3,3,i+1);plt.imshow(meas_p[idx,:,:,i])
    # plt.show()
    # pdb.set_trace()

    # the input of the network
    return meas, true, depths, vys, vxs

def data_augment_axial(name, array_dir, tof_cam):
    f = open('../FLAT/kinect/list/'+name+'.txt','r')
    message = f.read()
    files = message.split('\n')
    scenes = files[0:-1]
    scenes = [scene_n[0:-16]+'full/'+scene[-23:-7] for scene in scenes]
    idx = np.random.choice(len(scenes))
    scene_n = scenes[idx]

    print('Augmenting scene', scene_n)
    ## load all data
    # if the raw file does not exist, just find one and use
    if not os.path.exists(array_dir+scene_n[-16:]+'.pickle'):
        scenes = glob.glob(array_dir+'*.pickle')
        with open(scenes[0],'rb') as f:
            data = pickle.load(f)
        cam = data['cam']

        # separately read the true depth and true rendering
        with open(scene_n[0:-16]+'gt/'+scene_n[-16::],'rb') as f:
            gt=np.fromfile(f, dtype=np.float32)
        depth_true = np.reshape(gt,(cam['dimy']*4,cam['dimx']*4))

        with open(scene_n[0:-16]+'ideal/'+scene_n[-16::],'rb') as f:
            meas_gt=np.fromfile(f, dtype=np.int32)
        meas_gt = np.reshape(meas_gt,(cam['dimy'],cam['dimx'],9)).astype(np.float32)
    else:
        with open(array_dir+scene_n[-16::]+'.pickle','rb') as f:
            data = pickle.load(f)
        program = data['program']
        cam = data['cam']
        cam_t = data['cam_t']
        scene = data['scene']
        depth_true = data['depth_true']
        prop_idx = data['prop_idx']
        prop_s = data['prop_s'] 
        res_gt = tof_cam.process_gt_delay_vig_dist_surf_mapmax(cam, prop_idx, prop_s, scene, depth_true)
        meas_gt = res_gt['meas']

    # directly read pregenerate raw measurement
    with open(scene_n[0:-16]+'full/'+scene_n[-16::],'rb') as f:
        meas=np.fromfile(f, dtype=np.int32)
    meas = np.reshape(meas,(cam['dimy'],cam['dimx'],9)).astype(np.float32)
    msk = kinect_mask().astype(np.float32)
    meas = [meas[:,:,i]*msk/tof_cam.cam['map_max'] for i in range(meas.shape[-1])]
    meas_gt = [meas_gt[:,:,i]*msk/tof_cam.cam['map_max'] for i in range(meas_gt.shape[-1])]
    meas = np.stack(meas, -1)
    meas_gt = np.stack(meas_gt, -1)

    # reduce the resolution of the depth
    depth_true_s = scipy.misc.imresize(\
        depth_true,\
        meas.shape[0:2],\
        mode='F'\
    )
    depth_true_s = tof_cam.dist_to_depth(depth_true_s)

    # load the mask and classification
    with open(scene_n[0:-16]+'msk'+'/'+scene_n[-16:],'rb') as f:
        msk_array=np.fromfile(f, dtype=np.float32)
    msk_array = np.reshape(msk_array,(cam['dimy'],cam['dimx'],4))
    msk = {}
    msk['background'] = msk_array[:,:,0]
    msk['edge'] = msk_array[:,:,1]
    msk['noise'] = msk_array[:,:,2]
    msk['reflection'] = msk_array[:,:,3]

    # compute mask
    msk_true_s = msk['background'] * msk['edge']

    # true data
    d = np.expand_dims(depth_true_s,-1)
    m = np.expand_dims(msk_true_s, -1)
    true = np.concatenate([d, meas, np.zeros(meas.shape),np.zeros(meas.shape), meas_gt, m], -1)

    # cut the regions
    meas = meas[20:-20,:,:]
    true = true[20:-20,:,:]

    # fig = plt.figure()
    # for i in range(9):ax=fig.add_subplot(3,3,i+1);plt.imshow(meas[:,:,i])
    # fig = plt.figure()
    # for i in range(9):ax=fig.add_subplot(3,3,i+1);plt.imshow(meas_gt[20:-20,:,i])
    # fig = plt.figure();plt.imshow(depth_true_s)
    # plt.show()

    # # select a part of the image
    # py = int(meas.shape[0]/2)
    # px = int(meas.shape[1]/2)
    # meas_p = []
    # true_p = []
    # for iy in range(0,meas.shape[0]-py+1,int(py/2)):
    #     for ix in range(0,meas.shape[1]-px+1,int(px/2)):
    #         meas_p.append(meas[iy:(iy+py),ix:(ix+px)])
    #         true_p.append(true[iy:(iy+py),ix:(ix+px)])
    # meas_p = np.stack(meas_p, 0)
    # true_p = np.stack(true_p, 0)

    meas = np.expand_dims(meas, 0)
    true = np.expand_dims(true, 0)

    # the input of the network
    return meas, true

def leaky_relu(x):
    alpha = 0.1
    x_pos = tf.nn.relu(x)
    x_neg = tf.nn.relu(-x)
    return x_pos - alpha * x_neg

def flownet(x, flg=True):
    x_shape=[None, 384, 512, 9]
    output_shape = [None, 384, 512, 18]
    pref = 'flow_'

    # whether to train flag
    train_ae = flg

    # define initializer for the network
    keys = ['conv','upsample']
    keys_avoid = ['OptimizeLoss']
    inits = []

    from training_MOM import tof_net_func
    net_name = 'MOM'
    init_net = learn.Estimator(
        model_fn=tof_net_func,
        model_dir="./models/kinect/"+net_name,
    )
    if init_net != None:
        for name in init_net.get_variable_names():
            # select certain variables
            flag_init = False
            for key in keys:
                if key in name:
                    flag_init = True
            for key in keys_avoid:
                if key in name:
                    flag_init = False
            if flag_init:
                name_f = name.replace('/','_')
                num = str(init_net.get_variable_value(name).tolist())
                # self define the initializer function
                from tensorflow.python.framework import dtypes
                from tensorflow.python.ops.init_ops import Initializer
                exec("class "+name_f+"(Initializer):\n def __init__(self,dtype=tf.float32): self.dtype=dtype \n def __call__(self,shape,dtype=None,partition_info=None): return tf.cast(np.array("+num+"),dtype=self.dtype)\n def get_config(self):return {\"dtype\": self.dtype.name}")
                inits.append(name_f)

    # autoencoder
    n_input = x_shape[-1]
    n_filters=[18,32,32,64,64,128,128,256,256,256,256,512]
    filter_sizes=[None,7,5,5,5,5,3,3,3,3,3,3]
    pool_sizes=[None,2,1,2,1,2,1,2,1,2,1,2]
    pool_strides=[None,2,1,2,1,2,1,2,1,2,1,2]
    skips = [False,False,True,False,True,False,True,False,True,False,True,False]
    filter_sizes_skips = [7,5,5,5,5,3,3,3,3,3,3]

    n_filters_mix = [18,18,18,18]
    filter_sizes_mix=[1,3,3,3,3]

    # initializer 
    min_init = -1
    max_init = 1

    # change space
    ae_inputs = tf.identity(x,'ae_inputs')

    # prepare input
    current_input = tf.identity(ae_inputs, name="input")
    # convolutional layers: encoder
    conv = []
    pool = [current_input]
    for i in range(1,len(n_filters)):
        name = pref+"conv_"+str(i)

        # define the initializer 
        if name+'_bias' in inits:
            bias_init = eval(name+'_bias()')
        else:
            bias_init = tf.zeros_initializer()
        if name+'_kernel' in inits:
            kernel_init = eval(name+'_kernel()')
        else:
            kernel_init = tf.random_normal_initializer(0,1e-1)

        # convolution
        conv.append(\
            tf.layers.conv2d(\
                inputs=current_input,
                filters=n_filters[i],
                kernel_size=[filter_sizes[i], filter_sizes[i]],
                padding="same",
                activation=leaky_relu,
                trainable=train_ae,
                kernel_initializer=kernel_init,
                bias_initializer=bias_init,
                name=name,
            )
        )
        if pool_sizes[i] == 1 and pool_strides[i] == 1:
            pool.append(conv[-1])
        else:
            pool.append(\
                tf.layers.max_pooling2d(\
                    inputs=conv[-1],
                    pool_size=[pool_sizes[i],pool_sizes[i]],
                    strides=pool_strides[i],
                    name=pref+"pool_"+str(i)
                )
            )
        current_input = pool[-1]

    # convolutional layer: decoder
    # upsampling
    upsamp = []
    current_input = pool[-1]
    for i in range(len(n_filters)-1,0,-1):
        name = pref+"upsample_"+str(i-1)

        # define the initializer 
        if name+'_bias' in inits:
            bias_init = eval(name+'_bias()')
        else:
            bias_init = tf.zeros_initializer()
        if name+'_kernel' in inits:
            kernel_init = eval(name+'_kernel()')
        else:
            kernel_init = tf.random_normal_initializer(0,1e-1)

        # upsampling
        current_input = tf.layers.conv2d_transpose(\
            inputs=current_input,
            filters=n_filters[i-1],
            kernel_size=[filter_sizes[i], filter_sizes[i]],
            strides=(pool_strides[i], pool_strides[i]),
            padding="same",
            activation=leaky_relu,
            trainable=train_ae,
            kernel_initializer=kernel_init,
            bias_initializer=bias_init,
            name=name
        )

        # skip connection
        if skips[i-1] == True:
            name = pref+"skip_conv_"+str(i-1)

            # define the initializer 
            if name+'_bias' in inits:
                bias_init = eval(name+'_bias()')
            else:
                bias_init = tf.zeros_initializer()
            if name+'_kernel' in inits:
                kernel_init = eval(name+'_kernel()')
            else:
                kernel_init = tf.random_normal_initializer(0,1e-5)

            tmp = [current_input]
            tmp.append(pool[i-1])
            current_input = tf.concat(tmp, -1)
            current_input = tf.layers.conv2d(\
                inputs=current_input,
                filters=n_filters[i-1]*2,
                kernel_size=[filter_sizes_skips[i-1], filter_sizes_skips[i-1]],
                padding="same",
                activation=leaky_relu,
                trainable=train_ae,
                kernel_initializer=kernel_init,
                bias_initializer=bias_init,
                name=name,
            )
        upsamp.append(current_input)

    # mix
    mix = []
    for i in range(1,len(n_filters_mix)):
        name = pref+"mix_conv_"+str(i)

        # define the initializer 
        if name+'_bias' in inits:
            bias_init = eval(name+'_bias()')
        else:
            bias_init = tf.zeros_initializer()
        if name+'_kernel' in inits:
            kernel_init = eval(name+'_kernel()')
        else:
            kernel_init = tf.random_normal_initializer(0,1e-1)

        if i == (len(n_filters_mix)-1):
            activation=None
        else:
            activation=leaky_relu

        # convolution
        mix.append(\
            tf.layers.conv2d(\
                inputs=current_input,
                filters=n_filters_mix[i],
                kernel_size=[filter_sizes_mix[i], filter_sizes_mix[i]],
                padding="same",
                activation=activation,
                trainable=train_ae,
                kernel_initializer=kernel_init,
                bias_initializer=bias_init,
                name=name,
            )
        )
        current_input = mix[-1]

    v = tf.identity(current_input,name="v")

    return v

def get_values_at_coordinates(im, coord):
    input_as_vector = tf.reshape(im, [-1,tf.shape(im)[3]])
    coordinates_as_indices = (coord[:,:,0] * tf.shape(im)[2]) + coord[:,:,1]
    im_idx = tf.range(0, tf.shape(im)[0])* tf.shape(im)[1] * tf.shape(im)[2]
    im_idx = tf.expand_dims(im_idx, -1)
    coordinates_as_indices = coordinates_as_indices + im_idx
    coordinates_as_indices = tf.reshape(coordinates_as_indices, [-1])
    im_interp = tf.gather(input_as_vector, coordinates_as_indices,axis=0)
    im_interp = tf.reshape(im_interp,[-1, tf.shape(im)[1],tf.shape(im)[2],tf.shape(im)[3]])
    return im_interp

def interp2_bilinear(m,v):
    # derived from 
    # https://stackoverflow.com/questions/34902782/interpolated-sampling-of-points-in-an-image-with-tensorflow
    m_shape=[-1, 424, 512, 1]
    v_shape=[-1, 424*512, 2]

    top_left = tf.cast(tf.floor(v), tf.int32)

    top_right = tf.cast(
        tf.concat([tf.floor(v[:,:, 0:1]), tf.ceil(v[:, :, 1:2])],-1), tf.int32)

    bottom_left = tf.cast(
        tf.concat([tf.ceil(v[:,:, 0:1]), tf.floor(v[:, :, 1:2])],-1), tf.int32)

    bottom_right = tf.cast(tf.ceil(v), tf.int32)

    values_at_top_left = get_values_at_coordinates(m, top_left)
    values_at_top_right = get_values_at_coordinates(m, top_right)
    values_at_bottom_left = get_values_at_coordinates(m, bottom_left)
    values_at_bottom_right = get_values_at_coordinates(m, bottom_right)

    horizontal_offset = v[:,:,0] - tf.cast(top_left[:,:,0], tf.float32)
    horizontal_offset = tf.reshape(horizontal_offset, [-1, tf.shape(m)[1], tf.shape(m)[2], 1])

    horizontal_interpolated_top = (
        ((1.0 - horizontal_offset) * values_at_top_left)
        + (horizontal_offset * values_at_top_right))

    horizontal_interpolated_bottom = (
        ((1.0 - horizontal_offset) * values_at_bottom_left)
        + (horizontal_offset * values_at_bottom_right))

    vertical_offset = v[:,:,1] - tf.cast(top_left[:,:,1], tf.float32)
    vertical_offset = tf.reshape(vertical_offset, [-1, tf.shape(m)[1], tf.shape(m)[2], 1])

    interpolated_result = (
        ((1.0 - vertical_offset) * horizontal_interpolated_top)
        + (vertical_offset * horizontal_interpolated_bottom))

    m_interp = tf.reshape(interpolated_result, tf.shape(m))

    return m_interp, \
        values_at_top_left, \
        values_at_top_right, \
        values_at_bottom_left, \
        values_at_bottom_right

def applyOpticalFlow(m,v):
    m_shape=[None, 424, 512, 1]
    v_shape=[None, 424, 512, 2]

    # 
    xx, yy = np.meshgrid(np.arange(int(v.shape[2])), np.arange(int(v.shape[1])))
    xx = tf.constant(xx.flatten(), dtype=dtype)
    yy = tf.constant(yy.flatten(), dtype=dtype)
    xx = tf.expand_dims(xx, 0)
    yy = tf.expand_dims(yy, 0)
    loc = tf.stack([yy,xx],-1)

    # 
    v = tf.reshape(v, [-1, tf.shape(v)[1]*tf.shape(v)[2], 2])

    # old location of the interpolated image
    loc = loc + v
    m_new, l, r, bl, br = interp2_bilinear(m, loc)

    return m_new, l, r, bl, br, loc

def mod_motion_blur_flownet(x, flg):
    # inputs 9 channel raw measurements, float32
    # outputs 9 channel raw measurements, float32
    x_shape = [-1, 384, 512, 9]
    y_shape = [-1, 384, 512, 9]

    # predict the flow, and apply the flow
    v = flownet(x,flg)
    v = -v
    x_warped = []
    for i in range(9):
        x_warped.append(applyOpticalFlow(x[:,:,:,i:i+1], v[:,:,:,i::9])[0])
    x_warped = tf.concat(x_warped,-1)

    return x_warped, v

def kpn(x, flg):
    x_shape=[None, 424, 512, 9]
    y_shape=[None, 424, 512, 1*1*9*9+9]
    pref = 'kpn_'

    # whether to train flag
    train_ae = flg

    # define initializer for the network
    keys = ['conv','upsample']
    keys_avoid = ['OptimizeLoss']
    inits = []

    init_net = None
    if init_net != None:
        for name in init_net.get_variable_names():
            # select certain variables
            flag_init = False
            for key in keys:
                if key in name:
                    flag_init = True
            for key in keys_avoid:
                if key in name:
                    flag_init = False
            if flag_init:
                name_f = name.replace('/','_')
                num = str(init_net.get_variable_value(name).tolist())
                # self define the initializer function
                from tensorflow.python.framework import dtypes
                from tensorflow.python.ops.init_ops import Initializer
                exec("class "+name_f+"(Initializer):\n def __init__(self,dtype=tf.float32): self.dtype=dtype \n def __call__(self,shape,dtype=None,partition_info=None): return tf.cast(np.array("+num+"),dtype=self.dtype)\n def get_config(self):return {\"dtype\": self.dtype.name}")
                inits.append(name_f)

    # autoencoder
    n_filters=[\
        64,\
        64,64,64,
        128,128,128,
        256,256,256,
        512,
    ]
    filter_sizes=[\
        None,
        7,5,5,
        5,3,3,
        3,3,3,
        3,
    ]
    pool_sizes=[\
        None,
        2,1,1,
        2,1,1,
        2,1,1,
        2,
    ]
    pool_strides=[\
        None,
        2,1,1,
        2,1,1,
        2,1,1,
        2,
    ]
    skips = [\
        False,
        False,False,True,
        False,False,True,
        False,False,True,
        False,
    ]
    filter_sizes_skips = [\
        3,
        3,3,3,
        3,3,3,
        3,3,3,
        3,
    ]

    n_output = y_shape[-1]
    n_filters_mix = [n_output,n_output,n_output,n_output]
    filter_sizes_mix=[3,3,3,3]

    # initializer 
    min_init = -1
    max_init = 1

    # change space
    ae_inputs = tf.identity(x,name='ae_inputs')

    # prepare input
    current_input = tf.identity(ae_inputs, name="input")
    # convolutional layers: encoder
    conv = []
    pool = [current_input]
    for i in range(1,len(n_filters)):
        name = pref+"conv_"+str(i)

        # define the initializer 
        if name+'_bias' in inits:
            bias_init = eval(name+'_bias()')
        else:
            bias_init = tf.zeros_initializer()
        if name+'_kernel' in inits:
            kernel_init = eval(name+'_kernel()')
        else:
            kernel_init = None

        # convolution
        conv.append(\
            tf.layers.conv2d(\
                inputs=current_input,
                filters=n_filters[i],
                kernel_size=[filter_sizes[i], filter_sizes[i]],
                padding="same",
                activation=leaky_relu,
                trainable=train_ae,
                kernel_initializer=kernel_init,
                bias_initializer=bias_init,
                name=name,
            )
        )
        if pool_sizes[i] == 1 and pool_strides[i] == 1:
            pool.append(conv[-1])
        else:
            pool.append(\
                tf.layers.max_pooling2d(\
                    inputs=conv[-1],
                    pool_size=[pool_sizes[i],pool_sizes[i]],
                    strides=pool_strides[i],
                    name=pref+"pool_"+str(i)
                )
            )
        current_input = pool[-1]

    # convolutional layer: decoder
    # upsampling
    upsamp = []
    current_input = pool[-1]
    for i in range(len(n_filters)-1,0,-1):
        name = pref+"upsample_"+str(i-1)

        # define the initializer 
        if name+'_bias' in inits:
            bias_init = eval(name+'_bias()')
        else:
            bias_init = tf.zeros_initializer()
        if name+'_kernel' in inits:
            kernel_init = eval(name+'_kernel()')
        else:
            kernel_init = None

        # upsampling
        current_input = tf.layers.conv2d_transpose(\
            inputs=current_input,
            filters=n_filters[i-1],
            kernel_size=[filter_sizes[i], filter_sizes[i]],
            strides=(pool_strides[i], pool_strides[i]),
            padding="same",
            activation=leaky_relu,
            trainable=train_ae,
            kernel_initializer=kernel_init,
            bias_initializer=bias_init,
            name=name
        )

        # skip connection
        if skips[i-1] == True:
            name = pref+"skip_conv_"+str(i-1)

            # define the initializer 
            if name+'_bias' in inits:
                bias_init = eval(name+'_bias()')
            else:
                bias_init = tf.zeros_initializer()
            if name+'_kernel' in inits:
                kernel_init = eval(name+'_kernel()')
            else:
                kernel_init = None

            tmp = [current_input]
            tmp.append(pool[i-1])
            current_input = tf.concat(tmp, -1)
            current_input = tf.layers.conv2d(\
                inputs=current_input,
                filters=n_filters[i-1]*2,
                kernel_size=[filter_sizes_skips[i-1], filter_sizes_skips[i-1]],
                padding="same",
                activation=leaky_relu,
                trainable=train_ae,
                kernel_initializer=kernel_init,
                bias_initializer=bias_init,
                name=name,
            )
        upsamp.append(current_input)

    # mix
    mix = []
    for i in range(1,len(n_filters_mix)):
        name = pref+"mix_conv_"+str(i)

        # define the initializer 
        if name+'_bias' in inits:
            bias_init = eval(name+'_bias()')
        else:
            bias_init = tf.zeros_initializer()
        if name+'_kernel' in inits:
            kernel_init = eval(name+'_kernel()')
        else:
            kernel_init = None

        if i == (len(n_filters_mix)-1):
            activation=None
        else:
            activation=leaky_relu

        # convolution
        mix.append(\
            tf.layers.conv2d(\
                inputs=current_input,
                filters=n_filters_mix[i],
                kernel_size=[filter_sizes_mix[i], filter_sizes_mix[i]],
                padding="same",
                activation=activation,
                trainable=train_ae,
                kernel_initializer=kernel_init,
                bias_initializer=bias_init,
                name=name,
            )
        )
        current_input = mix[-1]

    ae_outputs = tf.identity(current_input,name="ae_output")
    return ae_outputs

def mod_multi_reflection_kpn(x, flg):
    # inputs 9 channel raw measurements, float32
    # outputs 9 channel raw measurements, float32
    x_shape = [-1, 384, 512, 9]
    y_shape = [-1, 384, 512, 9]
    k_shape = [3,3]

    output = kpn(x, flg)
    biass = output[:,:,:,-9::]
    kers = output[:,:,:,0:-9]
    kers = tf.reshape(kers,[-1, tf.shape(x)[1], tf.shape(x)[2], 9, 9])

    # kernel
    k = np.zeros([k_shape[0]*k_shape[1],1,k_shape[0]*k_shape[1]])
    for idx in range(k_shape[0]*k_shape[1]):k[idx,0,idx] = 1
    k = np.reshape(k, [k_shape[0],k_shape[1],1,k_shape[0]*k_shape[1]])
    k = k.astype(np.float32)

    ims = []
    for i in range(9):
        im = x[:,:,:,i:i+1]
        im_out = tf.nn.conv2d(im, k, strides=[1,1,1,1], padding='SAME')
        im_out = im_out * kers[:,:,:,:,i]
        ims.append(tf.reduce_sum(im_out, -1))
    ims = tf.stack(ims, -1)
    # ims = ims + biass

    return ims, output

def depth_prediction_mlp(x, flg):
    x_shape=[None, 424, 512, 9]
    y_shape=[None, 424, 512, 2]
    pref = 'depth_'

    # whether to train flag
    train_ae = flg

    # define initializer for the network
    keys = ['freq']
    keys_avoid = ['OptimizeLoss']
    inits = []
    from training import init_net
    if init_net != None:
        for name in init_net.get_variable_names():
            # select certain variables
            flag_init = False
            for key in keys:
                if key in name:
                    flag_init = True
            for key in keys_avoid:
                if key in name:
                    flag_init = False
            if flag_init:
                name_f = name.replace('/','_')
                num = str(init_net.get_variable_value(name).tolist())
                # self define the initializer function
                from tensorflow.python.framework import dtypes
                from tensorflow.python.ops.init_ops import Initializer
                exec("class "+name_f+"(Initializer):\n def __init__(self,dtype=tf.float32): self.dtype=dtype \n def __call__(self,shape,dtype=None,partition_info=None): return tf.cast(np.array("+num+"),dtype=self.dtype)\n def get_config(self):return {\"dtype\": self.dtype.name}")
                inits.append(name_f)

    # autoencoder
    
    n_filters=[\
        None,\
        81,81,81,2
    ]
    filter_sizes=[\
        None,
        3,1,1,1,
    ]

    # initializer 
    min_init = -1
    max_init = 1

    # change space
    ae_inputs = tf.identity(x,'ae_inputs')

    # prepare input
    current_input = tf.identity(ae_inputs, name="input")
    # convolutional layers: encoder
    conv = []
    for i in range(1,len(n_filters)):
        name = "conv_"+str(i)

        # define the initializer 
        if name+'_bias' in inits:
            bias_init = eval(name+'_bias()')
        else:
            bias_init = tf.zeros_initializer()
        if name+'_kernel' in inits:
            kernel_init = eval(name+'_kernel()')
        else:
            kernel_init = None

        # convolution
        conv.append(\
            tf.layers.conv2d(\
                inputs=current_input,
                filters=n_filters[i],
                kernel_size=[filter_sizes[i], filter_sizes[i]],
                padding="same",
                activation=leaky_relu,
                trainable=train_ae,
                kernel_initializer=kernel_init,
                bias_initializer=bias_init,
                name=pref+name,
            )
        )
        current_input = conv[-1]

    ae_outputs = tf.identity(current_input,name="ae_output")
    return ae_outputs

def mod_depth_prediction_mlp(x, flg):
    # inputs 9 channel raw measurements, float32
    # outputs 9 channel raw measurements, float32
    x_shape = [-1, 384, 512, 9]
    y_shape = [-1, 384, 512, 2]

    outputs = depth_prediction_mlp(x, flg)
    depth = outputs[:,:,:,0]
    conf = outputs[:,:,:,1]

    # 
    return depth, conf    

def processPixelStage1(m):
    # m is (None,424, 512, 9)
    # the first three is the first frequency
    tmp = []
    tmp.append(processMeasurementTriple(m[:,:,:,0:3], prms['ab_multiplier_per_frq'][0], trig_table0))
    tmp.append(processMeasurementTriple(m[:,:,:,3:6], prms['ab_multiplier_per_frq'][1], trig_table1))
    tmp.append(processMeasurementTriple(m[:,:,:,6:9], prms['ab_multiplier_per_frq'][2], trig_table2))

    m_out = [\
        tmp[0][:,:,:,0],tmp[1][:,:,:,0],tmp[2][:,:,:,0],
        tmp[0][:,:,:,1],tmp[1][:,:,:,1],tmp[2][:,:,:,1],
        tmp[0][:,:,:,2],tmp[1][:,:,:,2],tmp[2][:,:,:,2],
    ]
    m_out = tf.stack(m_out, -1)

    # return processMeasurementTriple(m[:,:,:,0:3], prms['ab_multiplier_per_frq'][0], trig_table0)
    return m_out

def processPixelStage1_mat(m):
    # if not saturated
    cos_tmp0 = np.stack([trig_table0[:,:,0],trig_table1[:,:,0],trig_table2[:,:,0]],-1)
    cos_tmp1 = np.stack([trig_table0[:,:,1],trig_table1[:,:,1],trig_table2[:,:,1]],-1)
    cos_tmp2 = np.stack([trig_table0[:,:,2],trig_table1[:,:,2],trig_table2[:,:,2]],-1)

    sin_negtmp0 = np.stack([trig_table0[:,:,3],trig_table1[:,:,3],trig_table2[:,:,3]],-1)
    sin_negtmp1 = np.stack([trig_table0[:,:,4],trig_table1[:,:,4],trig_table2[:,:,4]],-1)
    sin_negtmp2 = np.stack([trig_table0[:,:,5],trig_table1[:,:,5],trig_table2[:,:,5]],-1)

    # stack
    cos_tmp0 = np.expand_dims(cos_tmp0,0)
    cos_tmp1 = np.expand_dims(cos_tmp1,0)
    cos_tmp2 = np.expand_dims(cos_tmp2,0)
    sin_negtmp0 = np.expand_dims(sin_negtmp0,0)
    sin_negtmp1 = np.expand_dims(sin_negtmp1,0)
    sin_negtmp2 = np.expand_dims(sin_negtmp2,0)

    # 
    abMultiplierPerFrq = np.expand_dims(np.expand_dims(np.expand_dims(prms['ab_multiplier_per_frq'],0),0),0)

    ir_image_a = cos_tmp0 * m[:,:,:,0::3] + cos_tmp1 * m[:,:,:,1::3] + cos_tmp2 * m[:,:,:,2::3]
    ir_image_b = sin_negtmp0 * m[:,:,:,0::3] + sin_negtmp1 * m[:,:,:,1::3] + sin_negtmp2 * m[:,:,:,2::3]

    ir_image_a *= abMultiplierPerFrq
    ir_image_b *= abMultiplierPerFrq
    ir_amplitude = tf.sqrt(ir_image_a **2 + ir_image_b **2) * prms['ab_multiplier']

    # m_out = tf.concat([ir_image_a, ir_image_b, ir_amplitude], -1)
    # m_out = tf.tile(ir_image_a, [1,1,1,3])
    # m_out = tf.stack([ir_image_a, ir_image_b, ir_amplitude], -1)
    # m_out = tf.reshape(m_out, [-1, 424, 512, 9])

    return ir_image_a, ir_image_b, ir_amplitude

def processMeasurementTriple(m, abMultiplierPerFrq, trig_table):
    # m is (None,424,512,3)
    zmultiplier = tf.constant(z_table, dtype=dtype)

    # judge where saturation happens
    saturated = tf.cast(tf.less(tf.abs(m),1.0),dtype=dtype)
    saturated = 1 - saturated[:,:,:,0] * saturated[:,:,:,1] * saturated[:,:,:,2]

    # if not saturated
    cos_tmp0 = trig_table[:,:,0]
    cos_tmp1 = trig_table[:,:,1]
    cos_tmp2 = trig_table[:,:,2]

    sin_negtmp0 = trig_table[:,:,3]
    sin_negtmp1 = trig_table[:,:,4]
    sin_negtmp2 = trig_table[:,:,5]

    # stack
    cos_tmp0 = np.expand_dims(cos_tmp0,0)
    cos_tmp1 = np.expand_dims(cos_tmp1,0)
    cos_tmp2 = np.expand_dims(cos_tmp2,0)
    sin_negtmp0 = np.expand_dims(sin_negtmp0,0)
    sin_negtmp1 = np.expand_dims(sin_negtmp1,0)
    sin_negtmp2 = np.expand_dims(sin_negtmp2,0)

    ir_image_a = cos_tmp0 * m[:,:,:,0] + cos_tmp1 * m[:,:,:,1] + cos_tmp2 * m[:,:,:,2]
    ir_image_b = sin_negtmp0 * m[:,:,:,0] + sin_negtmp1 * m[:,:,:,1] + sin_negtmp2 * m[:,:,:,2]

    ir_image_a *= abMultiplierPerFrq
    ir_image_b *= abMultiplierPerFrq

    ir_amplitude = tf.sqrt(ir_image_a **2 + ir_image_b **2) * prms['ab_multiplier']

    m_out = tf.stack([ir_image_a, ir_image_b, ir_amplitude], -1)

    # # mask out the saturated pixel
    # zero_mat = tf.zeros(tf.shape(ir_image_a))
    # full_mat = tf.ones(tf.shape(ir_amplitude))
    # m_out_sat = tf.stack([zero_mat, zero_mat, full_mat], -1)
    # saturated = tf.expand_dims(saturated,-1)
    # m_out = saturated * m_out_sat + (1 - saturated) * m_out

    return m_out

def processPixelStage2(ira, irb, iramp):
    # m is (None, 424, 512, 9)
    # the first three is measurement a
    # the second three is measurement b
    # the third three is amplitude
    ratio = 100
    tmp0 = tf.atan2(ratio*(irb+1e-10), ratio*(ira+1e-10))
    flg = tf.cast(tf.less(tmp0,0.0), dtype)
    tmp0 = flg * (tmp0 + PI * 2) + (1 - flg) * tmp0

    tmp1 = tf.sqrt(ira**2 + irb**2) * prms['ab_multiplier']

    ir_sum = tf.reduce_sum(tmp1,-1)

    # disable disambiguation
    ir_min = tf.reduce_min(tmp1,-1)

    # phase mask
    phase_msk1 = tf.cast(\
        tf.greater(ir_min, prms['individual_ab_threshold']),
        dtype=dtype
    )
    phase_msk2 = tf.cast(\
        tf.greater(ir_sum, prms['ab_threshold']),
        dtype=dtype
    )
    phase_msk_t = phase_msk1 * phase_msk2

    # compute phase
    t0 = tmp0[:,:,:,0] / (2.0 * PI) * 3.0
    t1 = tmp0[:,:,:,1] / (2.0 * PI) * 15.0
    t2 = tmp0[:,:,:,2] / (2.0 * PI) * 2.0

    t5 = tf.floor((t1 - t0) * 0.3333333 + 0.5) * 3.0 + t0
    t3 = t5 - t2
    t4 = t3 * 2.0

    c1 = tf.cast(tf.greater(t4, -t4), dtype=dtype)
    f1 = c1 * 2.0 + (1 - c1) * (-2.0)
    f2 = c1 * 0.5 + (1 - c1) * (-0.5)
    t3 = t3 * f2
    t3 = (t3 - tf.floor(t3)) * f1

    c2 = tf.cast(tf.less(0.5,tf.abs(t3)), dtype=dtype) * \
        tf.cast(tf.less(tf.abs(t3),1.5), dtype=dtype)
    t6 = c2 * (t5 + 15.0) + (1 - c2) * t5
    t7 = c2 * (t1 + 15.0) + (1 - c2) * t1
    t8 = (tf.floor((t6-t2) * 0.5 + 0.5) * 2.0 + t2) * 0.5

    t6 /= 3.0
    t7 /= 15.0

    # transformed phase measurements (they are transformed and divided 
    # by the values the original values were multiplied with)
    t9 = t8 + t6 + t7 
    t10 = t9 / 3.0 # some avg

    t6 = t6 * 2.0 * PI
    t7 = t7 * 2.0 * PI
    t8 = t8 * 2.0 * PI

    t8_new = t7 * 0.826977 - t8 * 0.110264
    t6_new = t8 * 0.551318 - t6 * 0.826977
    t7_new = t6 * 0.110264 - t7 * 0.551318

    t8 = t8_new
    t6 = t6_new
    t7 = t7_new

    norm = t8**2 + t6**2 + t7**2
    mask = tf.cast(tf.greater(t9, 0.0), dtype)
    t10 = t10

    slope_positive = float(0 < prms['ab_confidence_slope'])

    ir_min_ = tf.reduce_min(tmp1,-1)
    ir_max_ = tf.reduce_max(tmp1,-1)

    ir_x = slope_positive * ir_min_ + (1 - slope_positive) * ir_max_

    ir_x = tf.log(ir_x)
    ir_x = (ir_x*prms['ab_confidence_slope']*0.301030 + prms['ab_confidence_offset'])*3.321928
    ir_x = tf.exp(ir_x)
    ir_x = tf.maximum(prms['min_dealias_confidence'], ir_x)
    ir_x = tf.minimum(prms['max_dealias_confidence'], ir_x)
    ir_x = ir_x **2

    mask2 = tf.cast(tf.greater(ir_x, norm), dtype)

    t11 = t10

    mask3 = tf.cast(\
        tf.greater(prms['max_dealias_confidence']**2, norm), 
        dtype
    )
    t10 = t10
    phase = t11

    # mask out dim regions
    phase = phase

    # phase to depth mapping
    zmultiplier = z_table
    xmultiplier = x_table

    phase_msk = tf.cast(tf.less(0.0, phase), dtype)
    phase = phase_msk*(phase+prms['phase_offset']) + (1-phase_msk)*phase

    depth_linear = zmultiplier * phase
    depth = depth_linear
    max_depth = phase * prms['unambiguous_dist'] * 2

    cond1 = tf.cast(tf.less(0.0,depth_linear), dtype) * \
        tf.cast(tf.less(0.0,max_depth), dtype)

    # xmultiplier = (xmultiplier * 90) / (max_depth**2 * 8192.0)

    # depth_fit = depth_linear / (-depth_linear * xmultiplier + 1)

    # depth_fit = tf.maximum(depth_fit, 0.0)
    # depth = cond1 * depth_fit + (1 - cond1) * depth_linear

    depth_out = depth
    ir_sum_out = ir_sum
    ir_out = tf.minimum(\
        tf.reduce_sum(iramp,-1)*0.33333333*prms['ab_output_multiplier'],
        65535.0
    )

    msk_out = cond1 * phase_msk_t * mask * mask2 * mask3
    return depth_out, ir_sum_out, ir_out, msk_out

def filterPixelStage1(m):
    # m is (None, 424, 512, 9)
    # the first three is measurement a
    # the second three is measurement b
    # the third three is amplitude

    # 
    norm2 = m[:,:,:,0:3]**2 + m[:,:,:,3:6]**2
    inv_norm = 1.0/tf.sqrt(norm2)

    # get rid of those nan
    inv_norm = tf.minimum(inv_norm,1e10)

    m_normalized = tf.stack([m[:,:,:,0:3] * inv_norm, m[:,:,:,3:6] * inv_norm], -1)

    threshold = prms['joint_bilateral_ab_threshold']**2 / prms['ab_multiplier']**2
    joint_bilateral_exp = prms['joint_bilateral_exp']
    threshold = tf.constant(threshold, dtype=dtype)
    joint_bilateral_exp = tf.constant(joint_bilateral_exp, dtype=dtype)

    # set the parts with norm2 < threshold to be zero
    norm_flag = tf.cast(tf.less(norm2, threshold), dtype = dtype)
    threshold = (1 - norm_flag) * threshold
    joint_bilateral_exp = (1- norm_flag) * joint_bilateral_exp

    # guided bilateral filtering
    gauss = prms['gaussian_kernel']
    weight_acc = tf.ones(tf.shape(m_normalized)[0:4])*gauss[1,1]
    weighted_m_acc0 = gauss[1,1] * m[:,:,:,0:3]
    weighted_m_acc1 = gauss[1,1] * m[:,:,:,3:6]

    # coefficient for bilateral space
    m_n = m_normalized

    # proxy for other m normalized
    m_l = tf.concat([m_n[:,:,1::,:],m_n[:,:,0:1,:]], 2)
    m_r = tf.concat([m_n[:,:,-1::,:],m_n[:,:,0:-1,:]],2)
    m_u = tf.concat([m_n[:,1::,:,:],m_n[:,0:1,:,:]],1)
    m_d = tf.concat([m_n[:,-1::,:,:],m_n[:,0:-1,:,:]],1)
    m_lu = tf.concat([m_l[:,1::,:,:],m_l[:,0:1,:,:]],1)
    m_ru = tf.concat([m_r[:,1::,:,:],m_r[:,0:1,:,:]],1)
    m_ld = tf.concat([m_l[:,-1::,:,:],m_l[:,0:-1,:,:]],1)
    m_rd = tf.concat([m_r[:,-1::,:,:],m_r[:,0:-1,:,:]],1)

    m_n_shift = [\
        m_rd, m_d, m_ld, m_r, m_l, m_ru, m_u, m_lu
    ]
    m_n_shift = tf.stack(m_n_shift, -1)

    # proxy of other_norm2
    norm2_l = tf.concat([norm2[:,:,1::,:],norm2[:,:,0:1,:]], 2)
    norm2_r = tf.concat([norm2[:,:,-1::,:],norm2[:,:,0:-1,:]],2)
    norm2_u = tf.concat([norm2[:,1::,:,:],norm2[:,0:1,:,:]],1)
    norm2_d = tf.concat([norm2[:,-1::,:,:],norm2[:,0:-1,:,:]],1)
    norm2_lu = tf.concat([norm2_l[:,1::,:,:],norm2_l[:,0:1,:,:]],1)
    norm2_ru = tf.concat([norm2_r[:,1::,:,:],norm2_r[:,0:1,:,:]],1)
    norm2_ld = tf.concat([norm2_l[:,-1::,:,:],norm2_l[:,0:-1,:,:]],1)
    norm2_rd = tf.concat([norm2_r[:,-1::,:,:],norm2_r[:,0:-1,:,:]],1)
    other_norm2 = tf.stack([\
        norm2_rd, norm2_d, norm2_ld, norm2_r,
        norm2_l, norm2_ru, norm2_u, norm2_lu,
    ],-1)

    dist = [\
        m_rd*m_n,m_d*m_n,m_ld*m_n,m_r*m_n,
        m_l*m_n,m_ru*m_n,m_u*m_n,m_lu*m_n,
    ]
    dist = -tf.reduce_sum(tf.stack(dist, -1),-2)
    dist += 1.0
    dist *= 0.5

    # color filtering
    gauss_f = gauss.flatten()
    gauss_f = np.delete(gauss_f, [4])
    joint_bilateral_exp = tf.tile(tf.expand_dims(joint_bilateral_exp,-1),[1,1,1,1,8])
    weight_f = tf.exp(-1.442695 * joint_bilateral_exp * dist)
    weight = tf.stack([gauss_f[k] * weight_f[:,:,:,:,k] for k in range(weight_f.shape[-1])],-1)

    # if (other_norm2 >= threshold)...
    threshold = tf.tile(tf.expand_dims(threshold,-1),[1,1,1,1,8])
    wgt_msk = tf.cast(tf.less(threshold,other_norm2),dtype=dtype)
    weight = wgt_msk * weight
    dist = wgt_msk * dist

    # coefficient for bilateral space
    ms = tf.stack([m[:,:,:,0:3], m[:,:,:,3:6]], -1)

    # proxy for other m normalized
    m_l = tf.concat([ms[:,:,1::,:],ms[:,:,0:1,:]], 2)
    m_r = tf.concat([ms[:,:,-1::,:],ms[:,:,0:-1,:]],2)
    m_u = tf.concat([ms[:,1::,:,:],ms[:,0:1,:,:]],1)
    m_d = tf.concat([ms[:,-1::,:,:],ms[:,0:-1,:,:]],1)
    m_lu = tf.concat([m_l[:,1::,:,:],m_l[:,0:1,:,:]],1)
    m_ru = tf.concat([m_r[:,1::,:,:],m_r[:,0:1,:,:]],1)
    m_ld = tf.concat([m_l[:,-1::,:,:],m_l[:,0:-1,:,:]],1)
    m_rd = tf.concat([m_r[:,-1::,:,:],m_r[:,0:-1,:,:]],1)
    m_shift = [\
        m_rd, m_d, m_ld, m_r, m_l, m_ru, m_u, m_lu
    ]
    m_shift = tf.stack(m_shift, -1)

    weighted_m_acc0 += tf.reduce_sum(weight * m_shift[:,:,:,:,0,:],-1)
    weighted_m_acc1 += tf.reduce_sum(weight * m_shift[:,:,:,:,1,:],-1)

    dist_acc = tf.reduce_sum(dist, -1)
    weight_acc += tf.reduce_sum(weight, -1)

    # test the edge
    bilateral_max_edge_test = tf.reduce_prod(tf.cast(\
        tf.less(dist_acc, prms['joint_bilateral_max_edge']),
        dtype
    ),-1)

    m_out = []
    wgt_acc_msk = tf.cast(tf.less(0.0,weight_acc),dtype=dtype)
    m_out.append(wgt_acc_msk * weighted_m_acc0 / weight_acc)
    m_out.append(wgt_acc_msk * weighted_m_acc1 / weight_acc)
    m_out.append(m[:,:,:,6:9])

    m_out = tf.concat(m_out, -1)

    # mask out the edge
    # do not filter the edge
    edge_step = 1
    edge_msk = np.zeros(m.shape[1:3])
    edge_msk[0:0+edge_step,:] = 1
    edge_msk[-1-edge_step+1::,:] = 1
    edge_msk[:,0:0+edge_step] = 1
    edge_msk[:,-1-edge_step+1::] = 1
    edge_msk = tf.constant(edge_msk, dtype=dtype)
    edge_msk = tf.tile(tf.expand_dims(tf.expand_dims(edge_msk,-1),0),[tf.shape(m)[0],1,1,9])

    m_out = edge_msk * m + (1-edge_msk) *m_out

    return m_out, bilateral_max_edge_test

def filterPixelStage2(raw_depth, raw_depth_edge, ir_sum):
    # raw depth is the raw depth prediction
    # raw_depth_edge is roughly the same as raw depth, except some part are zero if 
    # don't want to do edge filtering
    # mask out depth that is out of region
    depth_msk = tf.cast(tf.greater(raw_depth, prms['min_depth']), dtype) * \
        tf.cast(tf.less(raw_depth, prms['max_depth']), dtype)
    # mask out the edge
    # do not filter the edge of the image
    edge_step = 1
    edge_msk = np.zeros(raw_depth.shape[1:3])
    edge_msk[0:0+edge_step,:] = 1
    edge_msk[-1-edge_step+1::,:] = 1
    edge_msk[:,0:0+edge_step] = 1
    edge_msk[:,-1-edge_step+1::] = 1
    edge_msk = tf.constant(edge_msk, dtype=dtype)
    edge_msk = tf.tile(tf.expand_dims(edge_msk,0),[tf.shape(raw_depth)[0],1,1])

    # 
    knl = tf.constant(np.array([[1,1,1],[1,1,1],[1,1,1]]), dtype=dtype)
    knl = tf.expand_dims(tf.expand_dims(knl,-1),-1)
    ir_sum_exp = tf.expand_dims(ir_sum,-1)
    ir_sum_acc = tf.nn.conv2d(ir_sum_exp, knl, strides=[1,1,1,1], padding='SAME')
    squared_ir_sum_acc = tf.nn.conv2d(ir_sum_exp**2, knl, strides=[1,1,1,1], padding='SAME')
    ir_sum_acc = tf.squeeze(ir_sum_acc,-1)
    squared_ir_sum_acc = tf.squeeze(squared_ir_sum_acc,-1)
    min_depth = raw_depth
    max_depth = raw_depth

    # min_depth, max_depth
    m_n = raw_depth_edge 
    m_l = tf.concat([m_n[:,:,1::],m_n[:,:,0:1]], 2)
    m_r = tf.concat([m_n[:,:,-1::],m_n[:,:,0:-1]],2)
    m_u = tf.concat([m_n[:,1::,:],m_n[:,0:1,:]],1)
    m_d = tf.concat([m_n[:,-1::,:],m_n[:,0:-1,:]],1)
    m_lu = tf.concat([m_l[:,1::,:],m_l[:,0:1,:]],1)
    m_ru = tf.concat([m_r[:,1::,:],m_r[:,0:1,:]],1)
    m_ld = tf.concat([m_l[:,-1::,:],m_l[:,0:-1,:]],1)
    m_rd = tf.concat([m_r[:,-1::,:],m_r[:,0:-1,:]],1)
    m_shift = [\
        m_rd, m_d, m_ld, m_r, m_l, m_ru, m_u, m_lu
    ]
    m_shift = tf.stack(m_shift, -1)
    nonzero_msk = tf.cast(tf.greater(m_shift, 0.0), dtype=dtype)
    m_shift_min = nonzero_msk * m_shift + (1 - nonzero_msk) * 99999999999
    min_depth = tf.minimum(tf.reduce_min(m_shift_min,-1), min_depth)
    max_depth = tf.maximum(tf.reduce_max(m_shift,-1), max_depth)

    #
    tmp0 = tf.sqrt(squared_ir_sum_acc*9.0 - ir_sum_acc**2)/9.0
    edge_avg = tf.maximum(\
        ir_sum_acc/9.0, prms['edge_ab_avg_min_value']
    )
    tmp0 /= edge_avg

    #
    abs_min_diff = tf.abs(raw_depth - min_depth)
    abs_max_diff = tf.abs(raw_depth - max_depth)

    avg_diff = (abs_min_diff + abs_max_diff) * 0.5
    max_abs_diff = tf.maximum(abs_min_diff, abs_max_diff)

    cond0 = []
    cond0.append(tf.cast(tf.less(0.0,raw_depth),dtype))
    cond0.append(tf.cast(tf.greater_equal(tmp0, prms['edge_ab_std_dev_threshold']),dtype))
    cond0.append(tf.cast(tf.less(prms['edge_close_delta_threshold'], abs_min_diff),dtype))
    cond0.append(tf.cast(tf.less(prms['edge_far_delta_threshold'], abs_max_diff),dtype))
    cond0.append(tf.cast(tf.less(prms['edge_max_delta_threshold'], max_abs_diff),dtype))
    cond0.append(tf.cast(tf.less(prms['edge_avg_delta_threshold'], avg_diff),dtype))

    cond0 = tf.reduce_prod(tf.stack(cond0,-1),-1)

    depth_out = (1 - cond0) * raw_depth

    # !cond0 part 
    edge_test_msk = 1 - tf.cast(tf.equal(raw_depth_edge, 0.0),dtype)
    depth_out = raw_depth * (1 - cond0) * edge_test_msk

    # mask out the depth out of the range
    depth_out = depth_out * depth_msk

    # mask out the edge
    depth_out = edge_msk * raw_depth + (1 - edge_msk) * depth_out

    # msk_out 
    msk_out = edge_msk + (1 - edge_msk) * depth_msk * (1 - cond0) * edge_test_msk

    return depth_out, msk_out

def tof_net_func(x, y, mode):
    # it reverse engineer the kinect 
    x_shape=[-1, 384, 512, 9]
    y_shape=[-1, 384, 512, 38]
    k_shape=[9,9]
    l = 2
    lr = 1e-7

    # convert to the default data type
    x = tf.cast(x, dtype)

    # align the images
    motion_train_flg = False
    x_warped_v, v = mod_motion_blur_flownet(x[:,:,:,0:9], motion_train_flg)

    # denoising and de-multiple-reflection
    reflection_train_flg = True
    x_warped_r, ratio = mod_multi_reflection_kpn(x_warped_v, reflection_train_flg)

    ##################################################################
    ## Kinect Pipeline
    ##################################################################
    # make the size to be 424,512
    y_idx = int((424-int(x_warped_r.shape[1]))/2)
    x_idx = int((512-int(x_warped_r.shape[2]))/2)
    zero_mat = tf.zeros([tf.shape(x_warped_r)[0], y_idx, int(x_warped_r.shape[2]), 9])
    x_warped_r_exp = tf.concat([zero_mat, x_warped_r, zero_mat],1)
    zero_mat = tf.zeros([tf.shape(x_warped_r)[0], 424, x_idx, 9])
    x_warped_r_exp = tf.concat([zero_mat, x_warped_r_exp, zero_mat],2)

    # 
    msk = kinect_mask().astype(np.float32)
    msk = np.expand_dims(np.expand_dims(msk,0),-1)
    x_kinect = x_warped_r_exp * msk

    # final depth prediction: kinect pipeline
    ira, irb, iramp = processPixelStage1_mat(x_kinect)
    depth_outs, ir_sum_outs, ir_outs, msk_out1 = processPixelStage2(ira, irb, iramp)

    # creates the mask
    ms = tf.concat([ira, irb, iramp], -1)
    bilateral_max_edge_tests = filterPixelStage1(ms)[1]
    depth_out_edges = depth_outs * bilateral_max_edge_tests
    msk_out2 = filterPixelStage2(depth_outs, depth_out_edges, ir_outs)[1]
    msk_out3 = tf.cast(tf.greater(depth_outs, prms['min_depth']),dtype=dtype)
    msk_out4 = tf.cast(tf.less(depth_outs, prms['max_depth']),dtype=dtype)
    depth_msk = tf.cast(tf.greater(msk_out2*msk_out3*msk_out4, 0.5),dtype=dtype)

    depth_outs /= 1000.0
    depth_outs *= depth_msk

    # baseline correction
    depth_outs = depth_outs*base_cor['k'] +base_cor['b']

    # cut out the zero regions
    if x_idx == 0:
        depth_outs = depth_outs[:,y_idx:-y_idx,:]
        depth_msk = depth_msk[:,y_idx:-y_idx,:]
    else:
        depth_outs = depth_outs[:,y_idx:-y_idx,x_idx:-x_idx]
        depth_msk = depth_msk[:,y_idx:-y_idx,x_idx:-x_idx]

    # 
    loss = None
    train_op = None

    # compute loss (for TRAIN and EVAL modes)
    # fake a variable

    if mode != learn.ModeKeys.INFER:
        y = tf.cast(y, dtype)
        depth_true = y[:,:,:,0]
        x_warped_v_gt = y[:,:,:,1:10]
        v_gt = y[:,:,:,10:28]
        x_warped_r_gt = y[:,:,:,28:37]
        gt_mask = tf.cast(tf.greater(depth_true, 1e-4), dtype=dtype)*y[:,:,:,-1]
        depth_loss = (\
            tf.reduce_sum(\
                (tf.abs(depth_outs - depth_true)*depth_msk*gt_mask)**l
            )\
            /(tf.reduce_sum(depth_msk*gt_mask)+1e-4)
        )**(1/l)
        x_warped_v_loss = tf.reduce_mean(tf.abs(x_warped_v - x_warped_v_gt)**l)**(1/l)
        v_loss = tf.reduce_mean(tf.abs(v_gt - v)**l)**(1/l)
        x_warped_r_loss = tf.reduce_mean(tf.abs(x_warped_r - x_warped_r_gt)**l)**(1/l)

        # loss = 1e-6*depth_loss
        loss = depth_loss + 0*x_warped_r_loss
        # loss = 0*x_warped_v_loss + 0*v_loss + 0*x_warped_r_loss + 0*depth_loss
        loss = tf.identity(loss, name="loss")

        grad_loss = tf.reduce_sum(tf.abs(tf.gradients(tf.reduce_sum(depth_loss), depth_outs)))
        grad_loss = tf.identity(grad_loss, name='grad_loss')

        grad_depth = tf.reduce_sum(tf.abs(tf.gradients(tf.reduce_sum(depth_outs), ira)))
        grad_depth = tf.identity(grad_depth, name='grad_depth')

        # grad_ms = tf.reduce_sum(tf.abs(tf.gradients(tf.reduce_sum(ms[0,0,0,0]), x_kinect)))
        grad_ms = tf.reduce_sum(tf.abs(tf.gradients(ira, x_kinect)))
        grad_ms = tf.identity(grad_ms, name="grad_ms")

        grad_r = tf.reduce_sum(tf.abs(tf.gradients(tf.reduce_sum(x_kinect), x_warped_r)))
        grad_r = tf.identity(grad_r, name="grad_r")

        grad = tf.reduce_sum(tf.abs(tf.gradients(loss, x_warped_r)))
        grad = tf.identity(grad, name='grad')

        ms = tf.identity(tf.reduce_sum(ms), name='ms')

        # configure the training op (for TRAIN mode)
        if mode == learn.ModeKeys.TRAIN:
            train_op = tf.contrib.layers.optimize_loss(\
                loss=loss,
                global_step=tf.contrib.framework.get_global_step(),
                learning_rate=lr,
                optimizer="Adam"
            )

    # generate predictions
    predictions = {
        "depth": depth_outs,
    }
    # output intermediate things
    # ms = tf.identity(ms, name='ms')
    # x_tilde = tf.identity(x_tilde, name='x_tilde')
    x_warped_v = tf.identity(x_warped_v, name='x_warped_v')
    x_warped_r = tf.identity(x_warped_r, name='x_warped_r')
    v = tf.identity(v, name='v')
    ratio = tf.identity(ratio, name='ratio')
    depth_msk = tf.identity(depth_msk, name='depth_msk')
    msk_out1 = tf.identity(msk_out1, name='msk_out1')
    msk_out2 = tf.identity(msk_out2, name='msk_out2')
    tensors = [x_warped_v]+[x_warped_r]+[v]+[ratio]+[depth_msk]+[msk_out1]+[msk_out2]
    for tensor in tensors:
        predictions[tensor.name]=tensor

    # return a ModelFnOps object
    return model_fn_lib.ModelFnOps(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
    )

def training(trains, train_dir, vals, val_dir, tof_cam, tof_net, tr_num=1, batch_size=1, steps=500, iter_num=2000):
    # first prepare validation data
    x_val = []
    y_val = []
    val_num = 4
    for i in range(val_num):
        scenes = select_objects_val()
        x_t, y_t = data_augment_th(scenes, val_dir, tof_cam)[0:2]
        x_val.append(x_t)
        y_val.append(y_t)
    x_val = np.stack(x_val,0)
    y_val = np.stack(y_val,0)
    idx = np.random.choice(x_val.shape[0],val_num,replace=False)
    x_val = x_val[idx]
    y_val = y_val[idx]

    # data augmentation
    for i in range(iter_num):
        x = []
        y = []
        for i in range(tr_num):
            scenes = select_objects()
            x_t,y_t = data_augment_th(scenes, train_dir, tof_cam)[0:2]
            x.append(x_t)
            y.append(y_t)
        x = np.stack(x,0)
        y = np.stack(y,0)

        # set up logging for predictions
        tensors_to_log = {"Training loss": "loss"}
        logging_hook = tf.train.LoggingTensorHook(
            tensors=tensors_to_log, every_n_iter=1
        )
        validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
            x=x_val,
            y=y_val,
            every_n_steps=100,
            name='Validation',
        )

        # training
        tof_net.fit(\
            x=x,
            y=y,
            batch_size=batch_size,
            steps=steps,
            monitors=[logging_hook, validation_monitor]
        )

    return tof_net

def training_axial(trains, train_dir, vals, val_dir, tof_cam, tof_net, tr_num=1, batch_size=1, steps=500, iter_num=2000):
    # first prepare validation data
    x_val = []
    y_val = []
    val_num = 4
    for i in range(val_num):
        # randomly selects the validation data from two sets
        if np.random.uniform()> 0.5:
            scenes = select_objects_val()
            x_t, y_t = data_augment_th(scenes, val_dir, tof_cam)
        else:
            x_t, y_t = data_augment_axial('motion-real-validation', val_dir, tof_cam)
        x_val.append(x_t)
        y_val.append(y_t)
    x_val = np.concatenate(x_val,0)
    y_val = np.concatenate(y_val,0)
    idx = np.random.choice(x_val.shape[0],val_num,replace=False)
    x_val = x_val[idx]
    y_val = y_val[idx]

    # data augmentation
    for i in range(iter_num):
        x = []
        y = []
        for i in range(tr_num):
            # randomly selects the validation data from two sets
            if np.random.uniform()> 0.5:
                scenes = select_objects()
                x_t,y_t = data_augment_th(scenes, train_dir, tof_cam)
            else:
                x_t, y_t = data_augment_axial('motion-real-training1', val_dir, tof_cam)
            x.append(x_t)
            y.append(y_t)
        x = np.concatenate(x,0)
        y = np.concatenate(y,0)

        # set up logging for predictions
        tensors_to_log = {"Training loss": "loss"}
        logging_hook = tf.train.LoggingTensorHook(
            tensors=tensors_to_log, every_n_iter=1
        )
        validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
            x=x_val,
            y=y_val,
            every_n_steps=100,
            name='Validation',
        )

        # training
        tof_net.fit(\
            x=x,
            y=y,
            batch_size=batch_size,
            steps=steps,
            monitors=[logging_hook, validation_monitor]
        )

    return tof_net


if __name__ == '__main__':
    # data
    array_dir = '../FLAT/trans_render/static/'
    data_dir = '../FLAT/kinect/'

    # initialize the camera model
    tof_cam = kinect_real_tf()

    # input the folder that trains the data
    # only use the files listed
    f = open('../FLAT/kinect/list/train.txt','r')
    message = f.read()
    files = message.split('\n')
    trains = files[0:-1]
    trains = [data_dir+train for train in trains]

    f = open('../FLAT/kinect/list/test.txt','r')
    message = f.read()
    files = message.split('\n')
    vals = files[0:-1]
    vals = [data_dir+val for val in vals]
    vals = vals[0:2] # limit the validation set

    # create the network estimator for depth
    # thre means thresholding the multi-reflection indicator
    # dist means weighting the error based on true distance
    net_name = 'MOM_MRM_LF2'
    tof_net = learn.Estimator(
        model_fn=tof_net_func,
        model_dir="./models/kinect/"+net_name,
    )

    training(trains, array_dir, vals, array_dir, tof_cam, tof_net,\
             tr_num=2, batch_size=1, steps=200, iter_num=4000
    )

    # training_axial(trains, array_dir, vals, array_dir, tof_cam, tof_net,\
    #          tr_num=20, batch_size=2, steps=200, iter_num=4000
    # )
