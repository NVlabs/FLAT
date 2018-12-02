# license:  Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
#           Licensed under the CC BY-NC-SA 4.0 license
#           (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode). 

# this code simulate the approximate motion required
# all time unit are picoseconds (1 picosec = 1e-12 sec)
import sys
sys.path.insert(0,'../pipe/')
import numpy as np
import os, json, glob
import imageio
import matplotlib
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

def gen_approx_motion(scene_ns, array_dir, tof_cam, text_flg = False):
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
                # WARNING: IF YOU WANT TO USE TEXTURES
                # CREATE A DIRECTORY: 
                # ../FLAT/kinect/list/textures-curet/
                # PUT THE TEXTURE IMAGES (.png format) INTO IT
                
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
    ker = np.concatenate([kers_yf, kers_xf], -1)
    # normalize
    ker = ker / np.expand_dims(np.sum(ker, -1),-1) * 18

    pdb.set_trace()

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

    # # visualize the velocity
    # idx = 0
    # fig = plt.figure()
    # for i in range(9):ax = fig.add_subplot(3,3,i+1);plt.imshow(meas_p[idx,:,:,i])
    # plt.show()
    # pdb.set_trace()

    # the input of the network
    return meas, true, depths, vys, vxs


if __name__ == '__main__':
    # load the images
    array_dir = '../FLAT/trans_render/static/'
    data_dir = '../FLAT/kinect/'

    # input the folder that trains the data
    # only use the files listed
    f = open('../FLAT/kinect/list/test.txt','r')
    message = f.read()
    files = message.split('\n')
    tests = files[0:-1]
    if args.n_images!=-1:
        tests = tests[0:args.n_images]
    tests = [data_dir+test for test in tests]

    # initialize the camera model
    tof_cam = kinect_real_tf()

    # 
    num_of_scenes = 1
    num_of_loop = 1

    for i in range(len(num_of_loop)):
        idx = np.random.choice(len(tests), num_of_scenes)
        gen_approx_motion(tests[idx], array_dir, tof_cam, text_flg=False)

