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

    
    while(check==False):
        back_idx = np.random.choice(len(back_scenes),1,replace=False)
        fore_num = np.random.choice(1,1,replace=False)[0]+1
        fore_idx = np.random.choice(len(fore_scenes),fore_num,replace=False)

        # put the scenes together
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

def data_augment_th(scene_ns, array_dir, tof_cam, text_flg = False): # first loading each scene, and we will combine them then
    meass = []
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
        msk_true_s = msk['background'] * msk['edge']

        true = np.stack([depth_true_s,msk_true_s],2)
        true = np.concatenate([true, meas_gt], 2)

        # cut out some parts to make it dividable
        meas = meas[20:-20,:,:]
        depth_true_s = depth_true_s[20:-20,:]
        msk = msk_true_s[20:-20,:]

        if text_flg == True:
            # add textures (simply multiply a ratio)
            texts = glob.glob('../params/kinect/textures-curet/'+'*.png')
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

        # randomly generating an in image velocity
        v = np.random.rand(2)
        detv = np.random.uniform()*5 # the biggest velocity is 2 pixels per channel
        v = v / np.sqrt(np.sum(v**2)) * detv

        # randomly generating the 6 affine transform parameters
        max_pix = 10
        mov = 10
        while (np.abs(mov).max() >= max_pix):
            th1 = np.random.normal(0.0,0.02,[2,2])
            th1[0,0]+=1
            th1[1,1]+=1
            th2 = np.random.normal(0.0,1.0,[2,1])
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

        # append the data
        meass.append(meas)
        depths.append(depth_true_s)
        msks.append(msk)
        vs.append(th)

    # move the object and combine them by channel
    y = np.arange(meass[0].shape[0])
    x = np.arange(meass[0].shape[1])
    xx, yy = np.meshgrid(x,y)
    meass_new = []
    meass_old = []
    vys_new = []
    vxs_new = []
    msks_new = []
    depths_new = []

    mid = 4
    for i in range(9):
        meas_v = []
        depth_v = []
        msk_v = []
        vy_v = []
        vx_v = []
        meas_old_v = []
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

            f2 = scipy.interpolate.RegularGridInterpolator((y,x),depths[j],bounds_error=False, fill_value=0)
            depth_v.append(np.reshape(f2(pts), xx.shape))

            f3 = scipy.interpolate.RegularGridInterpolator((y,x),msks[j],bounds_error=False, fill_value=0)
            msk_v.append(np.reshape(f3(pts), xx.shape))

            vy_v.append(pts_y - yy)
            vx_v.append(pts_x - xx)

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
        vy_v  = np.stack(vy_v, -1)
        vx_v  = np.stack(vx_v, -1)

        # combine 
        depth_v[np.where(depth_v == 0)] = 999999999
        idx = np.argmin(depth_v, -1)
        pts = [yy.flatten(), xx.flatten(), idx.flatten()]
        meas_new = np.reshape(meas_v[pts], xx.shape)
        vy_new = np.reshape(vy_v[pts], xx.shape)
        vx_new = np.reshape(vx_v[pts], xx.shape)
        msk_new = np.reshape(msk_v[pts], xx.shape)
        meas_old = np.reshape(meas_old_v[pts], xx.shape)
        depth_new = np.reshape(depth_v[pts], xx.shape)
        depth_new[np.where(depth_new > 10)] = 0

        meass_new.append(meas_new)
        vys_new.append(vy_new)
        vxs_new.append(vx_new)
        msks_new.append(msk_new)
        depths_new.append(depth_new)

        # warp back the raw measurements to mask out invisible parts
        pts_y = (yy + vy_new).flatten()
        pts_x = (xx + vx_new).flatten()
        pts_y_fl = np.floor(pts_y).astype(np.int32)
        pts_y_cl = np.ceil(pts_y).astype(np.int32)
        pts_x_fl = np.floor(pts_x).astype(np.int32)
        pts_x_cl = np.ceil(pts_x).astype(np.int32)
        flg1 = pts_y_fl > 0
        flg2 = pts_y_cl < yy.shape[0]
        flg3 = pts_x_fl > 0
        flg4 = pts_x_cl < yy.shape[1]
        idx = np.where(flg1 * flg2 * flg3 * flg4)
        pts_y_fl = pts_y_fl[idx]
        pts_y_cl = pts_y_cl[idx]
        pts_x_fl = pts_x_fl[idx]
        pts_x_cl = pts_x_cl[idx]
        pts = (\
            np.concatenate([pts_y_fl, pts_y_fl, pts_y_cl, pts_y_cl],0),
            np.concatenate([pts_x_fl, pts_x_cl, pts_x_fl, pts_x_cl],0)
        )

        # old measurement
        msk_back = np.zeros(xx.shape)
        msk_back[pts] = 1
        meas_old = msk_back * meas_old
        meass_old.append(meas_old)

    # stack the final data
    meas = []
    true = []

    # # visualize the velocity
    # fig = plt.figure()
    # for i in range(9):ax = fig.add_subplot(3,3,i+1);plt.imshow(meass_new[i])
    # plt.show()
    
    meas = np.stack(meass_new+msks_new, -1)
    true = np.stack(vys_new+vxs_new, -1)
    meass_old = np.stack(meass_old, -1)
    depths_new = np.stack(depths_new, -1)

    # the input of the networka
    return meas, true, meass_old, depths_new

def leaky_relu(x):
    alpha = 0.1
    x_pos = tf.nn.relu(x)
    x_neg = tf.nn.relu(-x)
    return x_pos - alpha * x_neg

def dnnOpticalFlow(x):
    x_shape=[None, 384, 512, 9]
    output_shape = [None, 384, 512, 18]

    # whether to train flag
    train_ae = True

    # define initializer for the network
    keys = ['freq']
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
        name = "conv_"+str(i)

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
                    name="pool_"+str(i)
                )
            )
        current_input = pool[-1]

    # convolutional layer: decoder
    # upsampling
    upsamp = []
    current_input = pool[-1]
    for i in range(len(n_filters)-1,0,-1):
        name = "upsample_"+str(i-1)

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
            name = "skip_conv_"+str(i-1)

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
        name = "mix_conv_"+str(i)

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

def tof_net_func(x, y, mode):
    # it reverse engineer the kinect 
    x_shape=[-1, 384, 512, 9]
    y_shape=[-1, 384, 512, 18]
    l = 2
    lr = 1e-5

    # convert to the default data type
    msks = tf.tile(x[:,:,:,9::],[1,1,1,2])
    msks = tf.cast(msks, dtype)
    x = tf.cast(x[:,:,:,0:9], dtype)
    v = dnnOpticalFlow(x)
    # v = v * msks

    # 
    loss = None
    train_op = None

    # compute loss (for TRAIN and EVAL modes)
    if mode != learn.ModeKeys.INFER:
        y = tf.cast(y, dtype)
        v_true = y
        loss = (tf.reduce_mean(\
            tf.abs(\
                (v-v_true)\
            )**l)
        )**(1/l)
        # loss1 = tf.reduce_sum((inten0 * inten1))/tf.reduce_sum(tf.sqrt(inten0**2*inten1**2))
        # loss2 = tf.reduce_sum((inten2 * inten1))/tf.reduce_sum(tf.sqrt(inten2**2*inten1**2))
        loss = tf.identity(loss, name="loss")

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
        "v": v,
    }
    # output intermediate things
    # ms = tf.identity(ms, name='ms')
    # x_tilde = tf.identity(x_tilde, name='x_tilde')
    tensors = []
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
    val_num = 1
    for i in range(val_num):
        if np.mod(i,1)==0:
            scenes = select_objects()
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
            if np.mod(i,10)==0:
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

    f = open('../FLAT/kinect/list/val.txt','r')
    message = f.read()
    files = message.split('\n')
    vals = files[0:-1]
    vals = [data_dir+val for val in vals]
    vals = vals[0:5] # limit the validation set

    # create the network estimator for depth
    # thre means thresholding the multi-reflection indicator
    # dist means weighting the error based on true distance
    net_name = 'MOM'
    tof_net = learn.Estimator(
        model_fn=tof_net_func,
        model_dir="./models/kinect/"+net_name,
    )

    training(trains, array_dir, vals, array_dir, tof_cam, tof_net,\
             tr_num=5, batch_size=1, steps=200, iter_num=4000
    )