# license:  Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
#           Licensed under the CC BY-NC-SA 4.0 license
#           (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode). 

# this code runs teh MOM-LF2 pipeline
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
from copy import deepcopy
from joblib import Parallel, delayed
import multiprocessing
from kinect_spec import *
import cv2
from numpy import linalg as LA
from scipy import signal

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
tf.logging.set_verbosity(tf.logging.INFO)
from testing_MRM_LF2 import data_augment


from vis_flow import *
from kinect_init import *

import parser
import argparse

PI = 3.14159265358979323846
raw_depth_new = 0
flg = False

dtype = tf.float32

def metric_valid(depth, gt, msk):
    # compute mean absolute error on places where msk = 1
    msk /= np.sum(msk)
    return np.sum(np.abs(depth - gt)*msk)

def testing_real_motion(tests, array_dir, output_dir, tof_cam, tof_net):
    # testing
    errs = []
    errs_base = []
    errs_num_pix = []
    errs_total = []
    errs_base_total = []
    step =1
    for iter_idx in range(0,len(tests),step):
        te_idx = np.arange(iter_idx,min(iter_idx+step,len(tests)))
        x = []
        y = []
        z_gts = []
        msk_gts = []
        for i in range(len(te_idx)):
            x_te,y_te,z_gt,msk_gt = data_augment(tests[te_idx[i]], array_dir, tof_cam)
            x.append(x_te)
            y.append(y_te)
            z_gts.append(z_gt)
            msk_gts.append(msk_gt)
        x = np.stack(x,0)
        y = np.stack(y,0)
        z_gts = np.stack(z_gts, 0)
        msk_gts = np.stack(msk_gts, 0)

        # predict data
        data = list(tof_net.predict(x=x))
        mid = 4
        for j in range(len(data)):
            ims_warped = []
            for k in range(9):
                msk = np.ones(x[j,:,:,0].shape)
                v = np.stack([data[j]['v'][:,:,k],data[j]['v'][:,:,k+9]],-1)
                ims = np.stack([x[j,:,:,k],x[j,:,:,mid]],-1)

                # visualize optical flow
                im_v_pred = viz_flow(v[:,:,0],v[:,:,1])
                msk_exp = np.expand_dims(msk,-1)

                # expand the contour of v to solve the convex hull problem
                kl = np.ones([3,3])
                flg_int = signal.convolve2d(msk!=0, kl, mode='same')
                flg_edge = (flg_int!=0).astype(int)-(msk!=0).astype(int)

                v_convy = signal.convolve2d(v[:,:,0],kl, mode='same')
                v_convy /= flg_int
                v_edgey = v_convy * flg_edge
                v_edgey[np.isnan(v_edgey)] = 0

                v_convx = signal.convolve2d(v[:,:,1],kl, mode='same')
                v_convx /= flg_int
                v_edgex = v_convx * flg_edge
                v_edgex[np.isnan(v_edgex)] = 0

                v_edge = np.stack([v_edgey, v_edgex],-1)

                v = v_edge + v

                msk = (v[:,:,0] != 0)
                msk_exp = np.expand_dims(msk, -1)
                v *= msk_exp

                # warp the image
                xx,yy = np.meshgrid(np.arange(v.shape[1]),np.arange(v.shape[0]))
                pts_new = np.stack([yy.flatten(),xx.flatten()],-1)
                # xx *= msk.astype(np.int32)
                # yy *= msk.astype(np.int32)
                xx = xx.flatten()
                yy = yy.flatten()
                v_x = v[:,:,1].flatten()
                v_y = v[:,:,0].flatten()
                xx_new = xx + v_x
                yy_new = yy + v_y
                pts = np.stack([yy_new,xx_new],-1)
                vals = ims[:,:,0].flatten()
                
                im_warped = scipy.interpolate.griddata(pts,vals.flatten(),pts_new)
                im_warped = np.reshape(im_warped,x.shape[1:3])

                # put the two images together
                zero_mat = np.zeros([x.shape[1],x.shape[2]])
                x_or = np.stack([ims[:,:,0],zero_mat,ims[:,:,1]],-1)
                x_new = np.stack([im_warped,zero_mat,ims[:,:,1]],-1)
                or_zero = np.where(x_or == 0)
                new_zero = np.where(x_new == 0)
                vmin = np.nanmin(np.stack([x_or,x_new],-1))
                vmax = np.nanmax(np.stack([x_or,x_new],-1))
                
                # normalize
                x_or = (x_or - vmin)/(vmax-vmin)
                x_new = (x_new - vmin)/(vmax-vmin)
                x_or[or_zero] = 0
                x_new[new_zero] = 0
                x_or1 = (x_or * 255).astype(np.uint8)
                x_new1 = (x_new * 255).astype(np.uint8)

                # computing error on image with and without alignment
                err_wo = np.sum(np.abs(ims[:,:,0] - ims[:,:,1]))
                im_warped[np.isnan(im_warped)] = 0
                err_w = np.sum(np.abs(im_warped - ims[:,:,1]))
                ims_warped.append(im_warped)

                fig = plt.figure()
                fig.add_subplot(2,2,1)
                plt.imshow(x_or1)
                plt.axis('off')
                plt.title('Before align error: '+('%4f' % err_wo)+'.')
                fig.add_subplot(2,2,3)
                plt.imshow(x_new1)
                plt.axis('off')
                plt.title('After align error: '+('%4f' % err_w)+'.')
                fig.add_subplot(2,2,4)
                plt.imshow(im_v_pred)
                plt.axis('off')
                            
                name = int(np.random.uniform()*1e10)
                plt.savefig(\
                    output_dir+str(name)+'.png',
                    bbox_inches='tight',
                    dpi = 2*512,
                )

            x_warped = np.expand_dims(np.stack(ims_warped,-1),0)
            x_or = x[j,:,:,0:9]
            msk =  np.expand_dims(np.ones(x[j,:,:,0].shape),-1)
            depth = np.zeros(msk.shape)
            y_or = np.expand_dims(np.concatenate([depth,msk,x_or],-1),0)
            x_or = np.expand_dims(x_or,0)

            name = int(np.random.uniform()*1e10)
            fig=plt.figure()
            plt.suptitle("Original image")
            for i in range(9):
                ax=fig.add_subplot(3,3,i+1)
                plt.imshow(x_or[0,:,:,i])
                plt.axis('off')
            plt.savefig(\
                output_dir+str(name)+'.png',
                bbox_inches='tight',
                dpi = 2*512,
            )

            name = int(np.random.uniform()*1e10)
            fig=plt.figure()
            plt.suptitle("Warped image")
            for i in range(9):
                ax=fig.add_subplot(3,3,i+1)
                plt.imshow(x_warped[0,:,:,i])
                plt.axis('off')
            plt.savefig(\
                output_dir+str(name)+'.png',
                bbox_inches='tight',
                dpi = 2*512,
            )

            # add shape
            y_or = np.concatenate([np.zeros([1,20,512,11]),y_or,np.zeros([1,20,512,11])],1)
            x_or = np.concatenate([np.zeros([1,20,512,9]),x_or,np.zeros([1,20,512,9])],1)
            x_warped = np.concatenate([np.zeros([1,20,512,9]),x_warped,np.zeros([1,20,512,9])],1)
            y_or = y_or.astype(np.float32)
            x_or = x_or.astype(np.float32)
            x_warped = x_warped.astype(np.float32)

            # remask the raw measurement
            msk = kinect_mask().astype(np.float32)
            x_or = [x_or[0,:,:,i]*msk*tof_cam.cam['map_max'] for i in range(x_or.shape[-1])]
            x_or = np.expand_dims(np.stack(x_or,-1),0)
            x_warped = [x_warped[0,:,:,i]*msk*tof_cam.cam['map_max'] for i in range(x_warped.shape[-1])]
            x_warped = np.expand_dims(np.stack(x_warped,-1),0)

            # compute depth
            depth_or = list(raw_depth_new.predict(x=x_or))[0]['depth']
            depth = list(raw_depth_new.predict(x=x_warped))[0]['depth']
            z_gt = np.concatenate([np.zeros([20,512]),z_gts[j],np.zeros([20,512])],0)

            # 
            vmin = z_gt[np.where(z_gt>1e-4)].min()
            vmax = z_gt.max()
            fig = plt.figure()
            ax = fig.add_subplot(2,4,1)
            plt.imshow(depth_or,vmin=vmin,vmax=vmax)
            plt.axis('off')
            msk = depth_or > 0.5
            err = np.sum(np.abs(depth_or - z_gt)*msk)/np.sum(msk)
            plt.title('Original raw, err: '+'%.4f'%err+'m')

            ax = fig.add_subplot(2,4,2)
            plt.imshow((depth_or - z_gt)*msk, vmin=-0.1,vmax=0.1)
            plt.axis('off')

            ax = fig.add_subplot(2,4,3)
            plt.imshow(depth,vmin=vmin,vmax=vmax)
            plt.axis('off')
            msk = depth > 0.5
            err = np.sum(np.abs(depth - z_gt)*msk)/np.sum(msk)
            plt.title('Corrected raw, err: '+'%.4f'%err+'m')

            ax = fig.add_subplot(2,4,4)
            plt.imshow((depth - z_gt)*msk, vmin=-0.1,vmax=0.1)
            plt.axis('off')

            ax = fig.add_subplot(2,4,5)
            plt.imshow(z_gt,vmin=vmin,vmax=vmax)
            plt.axis('off')
            msk = z_gt > 0.5
            err = np.sum(np.abs(z_gt - z_gt)*msk)/np.sum(msk)
            plt.title('Ground truth raw, err: '+'%.4f'%err+'m')

            ax = fig.add_subplot(2,4,6)
            plt.imshow((z_gt - z_gt)*msk, vmin=-0.1,vmax=0.1)
            plt.axis('off')

            ax = fig.add_subplot(2,4,7)
            plt.imshow(z_gt,vmin=vmin,vmax=vmax)
            plt.axis('off')
            plt.title('Ground truth raw')
            plt.colorbar()

            ax = fig.add_subplot(2,4,8)
            plt.imshow((z_gt - z_gt)*msk, vmin=-0.1,vmax=0.1)
            plt.axis('off')
            plt.colorbar()         

            name = int(np.random.uniform()*1e10)
            plt.savefig(\
                output_dir+str(name)+'.png',
                bbox_inches='tight',
                dpi = 2*512,
            )

    return

if __name__ == '__main__':

    parser = argparse.ArgumentParser("testing MOM")
    parser.add_argument('-n', '--n-images', type=int, default = -1, help ='number of images to process; -1 to process all images')
    args = parser.parse_args()

    array_dir = '../FLAT/trans_render/static/'
    data_dir = '../FLAT/kinect/'

    # initialize the camera model
    tof_cam = kinect_real_tf()

    # input the folder that trains the data
    # only use the files listed
    f = open('../FLAT/kinect/list/test_dyn.txt','r')
    message = f.read()
    files = message.split('\n')
    tests = files[0:-1]
    if args.n_images!=-1:
        tests = tests[0:args.n_images]
    tests = [data_dir+test for test in tests]

    # create the network estimator
    file_name = 'MOM'
    from training_MOM import tof_net_func
    tof_net = learn.Estimator(
        model_fn=tof_net_func,
        model_dir="./models/kinect/"+file_name,
    )

    # create the network estimator
    baseline_name = 'LF2'
    from LF2 import tof_net_func
    raw_depth_new = learn.Estimator(
        model_fn=tof_net_func,
        model_dir="./models/kinect/"+baseline_name,
    )

    # create output folder
    output_dir = './results/'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output_dir += 'kinect/'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    folder_name = file_name 
    output_dir = output_dir + folder_name + '/'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    testing_real_motion(tests, array_dir, output_dir, tof_cam, tof_net)
