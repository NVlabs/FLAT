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
                with open(scene[0:-21]+'gt/'+scene[-16::],'rb') as f:
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
                with open(scene[0:-21]+'gt/'+scene[-16::],'rb') as f:
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

    meas = np.expand_dims(meas,0)
    true = np.expand_dims(true,0)

    # # visualize the velocity
    # idx = 0
    # fig = plt.figure()
    # for i in range(9):ax = fig.add_subplot(3,3,i+1);plt.imshow(meas_p[idx,:,:,i])
    # plt.show()
    # pdb.set_trace()

    # the input of the network
    return meas, true

def data_augment_th_MOM(scene_ns, test_dir, tof_cam):
    # first loading each scene, and we will combine them then
    meass = []
    depths = []
    msks = []
    vs = []
    for scene_n in scene_ns:
        print('Augmenting scene', scene_n)
        ## load all data
        # if the raw file does not exist, just find one and use
        if not os.path.exists(test_dir+scene_n[-16:]+'.pickle'):
            scenes = glob.glob(test_dir+'*.pickle')
            with open(scenes[0],'rb') as f:
                data = pickle.load(f)
            cam = data['cam']

            # separately read the true depth and true rendering
            with open(test_dir[0:-1]+'-gt/'+scene_n[-16::],'rb') as f:
                gt=np.fromfile(f, dtype=np.float32)
            depth_true = np.reshape(gt,(cam['dimy']*4,cam['dimx']*4))

            with open(test_dir[0:-1]+'-ideal/'+scene_n[-16::],'rb') as f:
                meas_gt=np.fromfile(f, dtype=np.int32)
            meas_gt = np.reshape(meas_gt,(cam['dimy'],cam['dimx'],9)).astype(np.float32)
        else:
            with open(test_dir+scene_n[-16::]+'.pickle','rb') as f:
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
        with open(scene_n,'rb') as f:
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
        with open(test_dir[0:-1]+'-msk'+'/'+scene_n[-16:],'rb') as f:
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

        # add textures (simply multiply a ratio)
        texts = glob.glob('./dataset/textures-curet/'+'*.png')
        idx = np.random.choice(len(texts),1,replace=False)[0]
        im_text = cv2.imread(texts[idx],0).astype(np.float32)
        im_text /= 255.
        lo = np.random.uniform(0,1) # random range
        hi = np.random.uniform(lo,1)
        im_text = im_text * (hi-lo) + lo
        im_text = scipy.misc.imresize(im_text,meas.shape[0:2],mode='F')
        im_text = np.expand_dims(im_text,-1)

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

        # apply the texture
        # meas = meas * im_text

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
    return meas, true

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

def data_augment_real_motion(scene_n, array_dir, tof_cam):
    print('Augmenting scene', scene_n)
    ## load all data
    # if the raw file does not exist, just find one and use
    if not os.path.exists(test_dir+scene_n[-16:]+'.pickle'):
        scenes = glob.glob(test_dir+'*.pickle')
        with open(scenes[0],'rb') as f:
            data = pickle.load(f)
        cam = data['cam']

        # separately read the true depth and true rendering
        with open(test_dir[0:-1]+'gt/'+scene_n[-16::],'rb') as f:
            gt=np.fromfile(f, dtype=np.float32)
        depth_true = np.reshape(gt,(cam['dimy']*4,cam['dimx']*4))

        with open(test_dir[0:-1]+'ideal/'+scene_n[-16::],'rb') as f:
            meas_gt=np.fromfile(f, dtype=np.int32)
        meas_gt = np.reshape(meas_gt,(cam['dimy'],cam['dimx'],9)).astype(np.float32)
    else:
        with open(test_dir+scene_n[-16::]+'.pickle','rb') as f:
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
    with open(scene_n,'rb') as f:
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
    with open(test_dir[0:-1]+'msk'+'/'+scene_n[-16:],'rb') as f:
        msk_array=np.fromfile(f, dtype=np.float32)
    msk_array = np.reshape(msk_array,(cam['dimy'],cam['dimx'],4))
    msk = {}
    msk['background'] = msk_array[:,:,0]
    msk['edge'] = msk_array[:,:,1]
    msk['noise'] = msk_array[:,:,2]
    msk['reflection'] = msk_array[:,:,3]

    # compute mask
    msk_true_s = msk['background'] * msk['edge']

    # true = np.stack([depth_true_s, msk_true_s],-1)
    true = np.concatenate([meas_gt, meas_gt], -1)

    # cut the regions
    meas = meas[20:-20,:,:]
    true = true[20:-20,:,:]
    depth_true_s = depth_true_s[20:-20,:]
    msk_true_s = msk_true_s[20:-20,:]

    # the input of the network
    return meas, true, meas, depth_true_s