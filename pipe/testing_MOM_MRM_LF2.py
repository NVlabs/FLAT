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
from vis_flow import *
from kinect_init import *
from testing_MRM_LF2 import data_augment

PI = 3.14159265358979323846

raw_depth_new = 0
flg = False
dtype = tf.float32

def metric_valid(depth, gt, msk):
    # compute mean absolute error on places where msk = 1
    msk /= np.sum(msk)
    return np.sum(np.abs(depth - gt)*msk)

def testing(tests, test_dir, output_dir, tof_cam, tof_net):
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

        # chooose from the data
        num = 1
        idx = np.random.choice(x.shape[0],num,replace=False)
        x = x[idx]
        y = y[idx]
        z_gts = z_gts[idx]

        # predict data
        data = list(tof_net.predict(x=x))
        mid = 4

        k_shape=[
            (43,43),
            (33,33),
            (23,23),
            (13,13),
            (3,3),
            (13,13),
            (23,23),
            (33,33),
            (43,43),
        ]
        
        for j in range(len(data)):
            im_warped_v = data[j]['x_warped_v:0']
            im_warped_r = data[j]['x_warped_r:0']
            vs = data[j]['v_1:0']
            ratio = data[j]['ratio:0']
            depth = data[j]['depth']
            depth_msk = data[j]['depth_msk:0']
            msk_out1 = data[j]['msk_out1:0']
            msk_out2 = data[j]['msk_out2:0']            

            im_warped_v1 = []
            for k in range(9):
                v = vs[:,:,k::9]
                im = x[j,:,:,k]

                # warp the image
                x1 = np.arange(v.shape[1])
                y1 = np.arange(v.shape[0])
                xx,yy = np.meshgrid(x1,y1)
                xx = xx.flatten()
                yy = yy.flatten()
                v_x = v[:,:,1].flatten()
                v_y = v[:,:,0].flatten()
                xx_new = xx + v_x
                yy_new = yy + v_y
                pts = np.stack([yy_new,xx_new],-1)         
                f1 = scipy.interpolate.RegularGridInterpolator((y1,x1),im,bounds_error=False, fill_value=0)
                im_warped_v1.append(np.reshape(f1(pts),im.shape))
            im_warped_v1 = np.stack(im_warped_v1,-1)

            fig = plt.figure()
            plt.suptitle('Original Raw')
            for i in range(9):
                ax=fig.add_subplot(3,3,i+1);
                plt.imshow(x[j,:,:,i])
                plt.axis('off')            

            name = int(np.random.uniform()*1e10)
            plt.savefig(\
                output_dir+str(name)+'.png',
                bbox_inches='tight',
                dpi = 2*512,
            )

            fig = plt.figure()
            plt.suptitle('Warped Raw')
            for i in range(9):
                ax=fig.add_subplot(3,3,i+1);
                plt.imshow(im_warped_v[:,:,i]);
                plt.axis('off')

            name = int(np.random.uniform()*1e10)
            plt.savefig(\
                output_dir+str(name)+'.png',
                bbox_inches='tight',
                dpi = 2*512,
            )

            fig = plt.figure()
            plt.suptitle('Multi-reflection Removal Raw')
            for i in range(9):
                ax=fig.add_subplot(3,3,i+1);
                plt.imshow(im_warped_r[:,:,i])
                plt.axis('off')

            name = int(np.random.uniform()*1e10)
            plt.savefig(\
                output_dir+str(name)+'.png',
                bbox_inches='tight',
                dpi = 2*512,
            )

            # use the kinect pipeline to produce depth
            xs = [x[j,:,:,:], im_warped_r, im_warped_v]
            msk = kinect_mask().astype(np.float32)
            msk_or = np.ones([384,512,1])
            depths = []
            for x_or in xs:
                y_or = np.concatenate([msk_or,msk_or,x_or],-1)
                x_or = np.concatenate([np.zeros([20,512,9]),x_or,np.zeros([20,512,9])],0)
                y_or = np.concatenate([np.zeros([20,512,11]),y_or,np.zeros([20,512,11])],0)
                x_or = [x_or[:,:,i]*msk*tof_cam.cam['map_max'] for i in range(x_or.shape[-1])]
                x_or = np.stack(x_or,-1)
                x_or = np.expand_dims(x_or,0)
                y_or = np.expand_dims(y_or,0)
                depths.append(list(raw_depth_new.predict(x=x_or))[0]['depth'])
            depths.append(
                np.concatenate([np.zeros([20,512]),y[j,:,:,0],np.zeros([20,512])],0)
            )

            depth_or = depths[0][20:-20,:]
            depth = depths[1][20:-20,:,]
            depth_v = depths[2][20:-20,:]
            z_gt = depths[3][20:-20,:]

            vmin = z_gt[np.where(z_gt>1e-4)].min()
            vmax = z_gt.max()
            fig = plt.figure()
            ax = fig.add_subplot(2,4,1)
            plt.imshow(depth_or,vmin=vmin,vmax=vmax)
            msk = depth_or > 0.5
            err = np.sum(np.abs(depth_or - z_gt)*msk)/np.sum(msk)
            plt.title('Original, err: '+'%.4f'%err+'m')
            plt.axis('off')

            ax = fig.add_subplot(2,4,2)
            plt.imshow((depth_or - z_gt)*msk, vmin=-0.1,vmax=0.1)
            plt.axis('off')

            ax = fig.add_subplot(2,4,3)
            plt.imshow(depth_v,vmin=vmin,vmax=vmax)
            msk = depth_v > 0.5
            err = np.sum(np.abs(depth_v - z_gt)*msk)/np.sum(msk)
            plt.title('FN, err: '+'%.4f'%err+'m')
            plt.axis('off')

            ax = fig.add_subplot(2,4,4)
            plt.imshow((depth_v - z_gt)*msk, vmin=-0.1,vmax=0.1)
            plt.axis('off')

            ax = fig.add_subplot(2,4,5)
            plt.imshow(depth,vmin=vmin,vmax=vmax)
            msk = depth > 0.5
            err = np.sum(np.abs(depth - z_gt)*msk)/np.sum(msk)
            plt.title('FN-KPN, err: '+'%.4f'%err+'m')
            plt.axis('off')

            ax = fig.add_subplot(2,4,6)
            plt.imshow((depth - z_gt)*msk, vmin=-0.1,vmax=0.1) 
            plt.axis('off')           

            ax = fig.add_subplot(2,4,7)
            plt.imshow(z_gt,vmin=vmin,vmax=vmax)
            msk = z_gt > 0.5
            err = np.sum(np.abs(z_gt - z_gt)*msk)/np.sum(msk)
            plt.title('True, err: '+'%.4f'%err+'m')
            plt.colorbar()
            plt.axis('off')

            ax = fig.add_subplot(2,4,8)
            plt.imshow((z_gt - z_gt)*msk, vmin=-0.1,vmax=0.1)
            plt.colorbar()         
            plt.axis('off')

            name = int(np.random.uniform()*1e10)
            plt.savefig(\
                output_dir+str(name)+'.png',
                bbox_inches='tight',
                dpi = 2*512,
            )

    return

if __name__ == '__main__':
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
    tests = [data_dir+test for test in tests]

    # create the network estimator
    file_name = 'MOM_MRM_LF2'
    from training_MOM_MRM_LF2 import tof_net_func
    tof_net = learn.Estimator(
        model_fn=tof_net_func,
        model_dir="./models/kinect/"+file_name,
    )

    # load the baseline method
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


    testing(tests, array_dir, output_dir, tof_cam, tof_net)