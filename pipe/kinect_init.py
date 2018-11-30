# license:  Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
#           Licensed under the CC BY-NC-SA 4.0 license
#           (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode). 

import numpy as np
import pickle

prms = {
    'ab_multiplier' : 0.66666687,
    'ab_multiplier_per_frq'    : np.array([1.32258105, 1.00000000, 1.61290300]),
    'ab_output_multiplier'    : 16.000000,
    'phase_in_rad'    :    np.array([0.000000, 2.09439492, 4.18878984]),
    'joint_bilateral_ab_threshold'    :    3.0000000,
    'joint_bilateral_max_edge'    :    2.5,
    'joint_bilateral_exp'    :    5.0,
    'gaussian_kernel'    :    np.array([[0.106997304,0.113109797,0.106997304],\
                            [0.113109797,0.119571596,0.113109797],\
                            [0.106997304,0.113109797,0.106997304]]),
    'phase_offset'        :    0.0,
    'unambiguous_dist'    :    2083.33301,
    'individual_ab_threshold'    :    3.0,
    'ab_threshold'    :    10.0,
    'ab_confidence_slope'    :    -0.533057809,
    'ab_confidence_offset'    :    0.769489408,
    'min_dealias_confidence':    0.349065900,
    'max_dealias_confidence':    0.610865295,
    'edge_ab_avg_min_value'    :    50.0,
    'edge_ab_std_dev_threshold'    :    0.05,
    'edge_close_delta_threshold':    50.0,
    'edge_far_delta_threshold'    :    30.0,
    'edge_max_delta_threshold'    :    100.0,
    'edge_avg_delta_threshold'    :    0.0,
    'max_edge_count'            :    5.0,
    'kde_sigma_sqr'                :    0.0239282232,
    'unwrapping_likelihood_scale'    :    2.0,
    'phase_confidence_scale'    :    3.0,
    'kde_threshold'                :    0.5,
    'kde_neigborhood_size'        :    5,
    'num_hyps'                    :    2,
    'min_depth'                    :    500.0,
    'max_depth'                    :    6000.0,
}

with open('../params/kinect/z_table','rb') as f:
    z_table = np.fromfile(f,dtype=np.float32)
    z_table = np.reshape(z_table, (424, 512))

with open('../params/kinect/x_table','rb') as f:
    x_table = np.fromfile(f,dtype=np.float32)
    x_table = np.reshape(x_table, (424, 512))

with open('../params/kinect/trig_table0','rb') as f:
    trig_table0 = np.fromfile(f,dtype=np.float32)
    trig_table0 = np.reshape(trig_table0, (424,512,6))

with open('../params/kinect/trig_table1','rb') as f:
    trig_table1 = np.fromfile(f,dtype=np.float32)
    trig_table1 = np.reshape(trig_table1, (424,512,6))

with open('../params/kinect/trig_table2','rb') as f:
    trig_table2 = np.fromfile(f,dtype=np.float32)
    trig_table2 = np.reshape(trig_table2, (424,512,6))

with open('../params/kinect/kinect_baseline_correct.pickle','rb') as f:
    base_cor = pickle.load(f)
