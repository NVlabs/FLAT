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
from testing_MRM_LF2 import data_augment

PI = 3.14159265358979323846
flg = False
dtype = tf.float32

def leaky_relu(x):
    alpha = 0.1
    x_pos = tf.nn.relu(x)
    x_neg = tf.nn.relu(-x)
    return x_pos - alpha * x_neg

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

    output = kpn(x, flg)
    biass = output[:,:,:,-9::]
    kers = output[:,:,:,0:-9]
    kers = tf.reshape(kers,[-1, tf.shape(x)[1], tf.shape(x)[2], 1*1*9, 9])

    # 
    x_new = []
    for i in range(9):
        ker = kers[:,:,:,:,i]
        bias = biass[:,:,:,i]
        # x_new.append(tf.reduce_sum(ker * x,-1)+bias)
        x_new.append(tf.reduce_sum(ker * x,-1))
    x_new = tf.stack(x_new, -1)

    return x_new, output

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
    y_shape=[-1, 384, 512, 11]
    k_shape=[9,9]
    idx = 5
    l = 2
    lr_flow = 1e-7
    lr = 1e-7

    # convert to the default data type
    x = tf.cast(x, dtype)
    y = tf.cast(y, dtype)

    # denoising and de-multiple-reflection
    reflection_train_flg = True
    x_warped_r, ratio = mod_multi_reflection_kpn(x[:,:,:,0:9], reflection_train_flg)

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
        depth_true = y[:,:,:,0]
        x_warped_r_gt = y[:,:,:,2:11]
        gt_mask = tf.cast(tf.greater(depth_true, 1e-4), dtype=dtype)*y[:,:,:,1]
        depth_loss = (\
            tf.reduce_sum(\
                (tf.abs(depth_outs - depth_true)*depth_msk*gt_mask)**l
            )\
            /(tf.reduce_sum(depth_msk*gt_mask)+1e-4)
        )**(1/l)
        x_warped_r_loss = tf.reduce_mean(tf.abs(x_warped_r - x_warped_r_gt)**l)**(1/l)

        loss = depth_loss + 0*x_warped_r_loss
        # loss = x_warped_r_loss
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
        "depth": depth_outs,
    }
    # output intermediate things
    # ms = tf.identity(ms, name='ms')
    # x_tilde = tf.identity(x_tilde, name='x_tilde')
    x_warped_r = tf.identity(x_warped_r, name='x_warped_r')
    ratio = tf.identity(ratio, name='ratio')
    depth_msk = tf.identity(depth_msk, name='depth_msk')
    msk_out1 = tf.identity(msk_out1, name='msk_out1')
    msk_out2 = tf.identity(msk_out2, name='msk_out2')
    tensors = [x_warped_r]+[ratio]+[depth_msk]+[msk_out1]+[msk_out2]
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
    for i in range(len(vals)):
        x_t, y_t = data_augment(vals[i], val_dir, tof_cam,text_flg = True)[0:2]
        x_val.append(x_t)
        y_val.append(y_t)
    x_val = np.stack(x_val,0)
    y_val = np.stack(y_val,0)


    # data augmentation
    for i in range(iter_num):
        indices = np.random.choice(len(trains),tr_num,replace=False)
        x = []
        y = []
        for i in indices:
            x_t,y_t = data_augment(trains[i], train_dir, tof_cam, text_flg = True)[0:2]
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

    f = open('../FLAT/kinect/list/test.txt','r')
    message = f.read()
    files = message.split('\n')
    vals = files[0:-1]
    vals = [data_dir+val for val in vals]
    vals = vals[0:4] # limit the validation set

    # create the network estimator for depth
    # thre means thresholding the multi-reflection indicator
    # dist means weighting the error based on true distance
    net_name = 'MRM_LF2'
    tof_net = learn.Estimator(
        model_fn=tof_net_func,
        model_dir="./models/kinect/"+net_name,
    )

    training(trains, array_dir, vals, array_dir, tof_cam, tof_net,\
             tr_num=5, batch_size=1, steps=100, iter_num=4000
    )