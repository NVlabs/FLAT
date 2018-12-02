# license:  Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
#           Licensed under the CC BY-NC-SA 4.0 license
#           (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode). 

# this code runs the MRM_LF2* pipeline
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
import parser
import argparse

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
tf.logging.set_verbosity(tf.logging.INFO)
from vis_flow import *
from kinect_init import *

PI = 3.14159265358979323846
raw_depth_new = 0
flg = False

dtype = tf.float32

def metric_valid(depth, gt, msk):
    # compute mean absolute error on places where msk = 1
    msk /= np.sum(msk)
    return np.sum(np.abs(depth - gt)*msk)

def data_augment(scene_n, array_dir, tof_cam, text_flg = False):
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

    # apply the texture whether one wants the texture or not
    if text_flg == True:
        # add textures (simply multiply a ratio)
        # theoretically one should first add texture then add the noise
        # but doing it this approximate way is faster
        texts = glob.glob('../params/kinect/textures-curet/'+'*.png')
        idx = np.random.choice(len(texts),1,replace=False)[0]
        im_text = cv2.imread(texts[idx],0).astype(np.float32)
        im_text /= 255.
        lo = np.random.uniform(0,1) # random range
        hi = np.random.uniform(lo,1)
        im_text = im_text * (hi-lo) + lo
        im_text = scipy.misc.imresize(im_text,meas.shape[0:2],mode='F')
        im_text = np.expand_dims(im_text,-1)

        meas = meas * im_text
        meas_gt = meas_gt * im_text

    true = np.stack([depth_true_s, msk_true_s],-1)
    true = np.concatenate([true, meas_gt], -1)

    # cut the regions
    meas = meas[20:-20,:,:]
    true = true[20:-20,:,:]
    depth_true_s = depth_true_s[20:-20,:]
    msk_true_s = msk_true_s[20:-20,:]

    # the input of the network
    return meas, true, depth_true_s, msk_true_s

def testing(tests, array_dir, output_dir, tof_cam, tof_net):
    # testing
    errs = []
    errs_base = []
    errs_num_pix = []
    pix_num_all = 0
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
            x_warped = np.expand_dims(data[j]['x_warped_r:0'],0)
            depth = data[j]['depth']
            depth_gt = y[j,:,:,0]
            msk = data[j]['depth_msk:0']*(depth_gt>1e-4)
            err_2norm = np.sqrt(np.sum(((depth-depth_gt)*msk)**2)/np.sum(msk))
            err_1norm = np.sum((depth-depth_gt)*msk)/np.sum(msk)

            err_warped = np.sqrt(np.mean(np.abs(x_warped - y[:,:,:,2::])**2))

            fig = plt.figure()
            plt.suptitle('Original Raw')
            for i in range(9):
                ax=fig.add_subplot(3,3,i+1);
                plt.imshow(x[j,:,:,i]);
                plt.axis('off')
            name = int(np.random.uniform()*1e10)
            plt.savefig(\
                output_dir+str(name)+'.png',
                bbox_inches='tight',
                dpi = 2*512,
            )

            fig = plt.figure()
            plt.suptitle('Raw after MRM')
            for i in range(9):
                ax=fig.add_subplot(3,3,i+1);
                plt.imshow(x_warped[j,:,:,i]);
                plt.axis('off')
            name = int(np.random.uniform()*1e10)
            plt.savefig(\
                output_dir+str(name)+'.png',
                bbox_inches='tight',
                dpi = 2*512,
            )

            fig = plt.figure()
            msk_sign = kinect_mask().astype(np.float32)[20:-20,:]
            plt.suptitle('Ground truth Raw')
            for i in range(9):
                ax=fig.add_subplot(3,3,i+1);
                plt.imshow(y[j,:,:,i+2]);
                plt.axis('off')
            name = int(np.random.uniform()*1e10)
            plt.savefig(\
                output_dir+str(name)+'.png',
                bbox_inches='tight',
                dpi = 2*512,
            )

            # use the kinect pipeline to produce depth
            xs = [x[j,:,:,:]]
            msk_sign = kinect_mask().astype(np.float32)
            msk_or = np.ones([384,512,1])
            depths = []
            for x_or in xs:
                y_or = np.concatenate([msk_or,msk_or,x_or],-1)
                x_or = np.concatenate([np.zeros([20,512,9]),x_or,np.zeros([20,512,9])],0)
                y_or = np.concatenate([np.zeros([20,512,11]),y_or,np.zeros([20,512,11])],0)
                x_or = [x_or[:,:,i]*msk_sign*tof_cam.cam['map_max'] for i in range(x_or.shape[-1])]
                x_or = np.stack(x_or,-1)
                x_or = np.expand_dims(x_or,0)
                y_or = np.expand_dims(y_or,0)
                depths.append(list(raw_depth_new.predict(x=x_or))[0]['depth'])
            
            depth_or = depths[0]
            depth_or = depth_or[20:-20,:]

            vmin=prms['min_depth']/1000
            vmax=prms['max_depth']/1000
            msk_gt = msk_gts[j]
            fig=plt.figure()
            ax=fig.add_subplot(2,4,1)
            msk_or = (depth_or>vmin)*(depth_gt>vmin)*msk_gt
            err = np.sum(np.abs(depth_or - depth_gt)*msk_or)/np.sum(msk_or)

            # record the error
            err_list = np.abs(depth_or - depth_gt)
            err_list = err_list[np.where(msk_or>0.999)]
            pix_num_all += len(depth_or.flatten())
            errs_base.append(err_list)
            
            plt.title("Original, err: "+'%.4f'%err+'m')
            plt.imshow(depth_or*msk_or,vmin=vmin,vmax=vmax)
            plt.axis('off')

            ax=fig.add_subplot(2,4,2)
            plt.imshow((depth_or-depth_gt)*msk_or,vmin=-0.1,vmax=0.1)
            plt.axis('off')

            ax=fig.add_subplot(2,4,3)
            msk = (depth>vmin)*(depth_gt>vmin)*msk_gt
            err = np.sum(np.abs(depth - depth_gt)*msk)/np.sum(msk)

            # record the error
            err_list = np.abs(depth - depth_gt)
            err_list = err_list[np.where(msk>0.999)]
            errs.append(err_list)

            plt.title("KPN, err: "+'%.4f'%err+'m')
            plt.imshow(depth*msk,vmin=vmin,vmax=vmax)
            plt.axis('off')

            ax=fig.add_subplot(2,4,4)
            plt.imshow((depth-depth_gt)*msk,vmin=-0.1,vmax=0.1)
            plt.axis('off')

            ax=fig.add_subplot(2,4,7)
            plt.title("True depth")
            plt.imshow(depth_gt*msk,vmin=vmin,vmax=vmax)
            plt.colorbar()
            plt.axis('off')

            ax=fig.add_subplot(2,4,8)
            plt.title("True depth")
            plt.imshow((depth_gt-depth_gt)*msk,vmin=-0.1,vmax=0.1)
            plt.axis('off')

            name = int(np.random.uniform()*1e10)
            plt.savefig(\
                output_dir+str(name)+'.png',
                bbox_inches='tight',
                dpi = 2*512,
            )

            text_file = open(output_dir+"err.txt", "w")
            text_file.write("Mean base err:    Density:    Mean net err:    Density: \n")
            text_file.write(\
                str(np.mean(np.concatenate(errs_base,0)))+"    "+\
                str(len(np.concatenate(errs_base))/pix_num_all)+"    "+\
                str(np.mean(np.concatenate(errs,0)))+"    "+\
                str(len(np.concatenate(errs))/pix_num_all)+"\n"
            )

    return

if __name__ == '__main__':

    parser = argparse.ArgumentParser("testing MRM LF2")
    parser.add_argument('-n', '--n-images', type=int, default = -1, help='number of images to process; -1 to process all the images')
    args = parser.parse_args()

    array_dir = '../FLAT/trans_render/static/'
    data_dir = '../FLAT/kinect/'

    # initialize the camera model
    tof_cam = kinect_real_tf()

    # input the folder that trains the data
    # only use the files listed
    f = open('../FLAT/kinect/list/test.txt','r')
    message = f.read()
    files = message.split('\n')
    tests = files[0:-1]
    if args.n_images!=-1:
	    tests = tests[0:args.n_images]
    tests = [data_dir+test for test in tests]

    # create the network estimator
    file_name = 'MRM_LF2'
    from training_MRM_LF2 import tof_net_func
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
