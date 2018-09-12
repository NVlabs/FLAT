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
from testing_MOM import select_objects, data_augment_th

PI = 3.14159265358979323846

dtype = tf.float32

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
    y = tf.cast(y, dtype)
    v = dnnOpticalFlow(x)
    # v = v * msks

    # 
    loss = None
    train_op = None

    # compute loss (for TRAIN and EVAL modes)
    if mode != learn.ModeKeys.INFER:
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