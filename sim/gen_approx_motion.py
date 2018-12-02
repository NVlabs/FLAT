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

from vis_flow import *
from kinect_init import *

PI = 3.14159265358979323846
raw_depth_new = 0
flg = False

dtype = tf.float32

def gen_approx_motion(scene_ns, array_dir, tof_cam, text_flg = False, do_vis = True):
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
                # WARNING: IF YOU WANT TO USE TEXTURES
                # CREATE A DIRECTORY: 
                # ../FLAT/kinect/list/textures-curet/
                # PUT THE TEXTURE IMAGES (.png format) INTO IT

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

                # cut off the regions outside 
                idx = np.where((y_p<(yy.shape[0]-1))*(y_p>0)*(x_p<(xx.shape[1]-1))*(x_p>0))
                y_pc = y_p[idx]
                x_pc = x_p[idx]

                # add a map of zeros
                zero_map = np.zeros(xx.shape)
                zero_map[(np.floor(y_pc).astype(np.int32),np.floor(x_pc).astype(np.int32))] = 1
                zero_map[(np.ceil(y_pc).astype(np.int32),np.floor(x_pc).astype(np.int32))] = 1
                zero_map[(np.floor(y_pc).astype(np.int32),np.ceil(x_pc).astype(np.int32))] = 1
                zero_map[(np.ceil(y_pc).astype(np.int32),np.ceil(x_pc).astype(np.int32))] = 1

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
    depth_true = depth_true_s[20:-20,:]
    v = np.stack([vys, vxs], -2)

    if do_vis:
        # visualization
        fig = plt.figure()
        ax = fig.add_subplot(1,3,1)
        plt.imshow(np.mean(np.abs(meas),-1))
        plt.title('scene')
        plt.axis('off')
        ax = fig.add_subplot(1,3,2)
        plt.imshow(depth_true)
        plt.title('depth')
        plt.axis('off')
        ax = fig.add_subplot(1,3,3)
        v_max = np.max(np.sqrt((v[:,:,0,0]**2 + v[:,:,1,0]**2)))
        plt.imshow(viz_flow(v[:,:,0,0],v[:,:,1,0], scaledown=v_max))
        plt.title('flow')
        plt.axis('off')
        plt.show()

    # the input of the network
    return meas, depth_true, v


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
    tests = [data_dir+test for test in tests]

    # initialize the camera model
    tof_cam = kinect_real_tf()

    # 
    num_of_scenes = 1
    num_of_loop = 1

    for i in range(num_of_loop):
        idx = np.random.choice(len(tests), num_of_scenes)
        meas, depth, v = gen_approx_motion([\
            tests[j] for j in idx], 
            array_dir, 
            tof_cam, 
            text_flg=False, 
            do_vis=True
        )

