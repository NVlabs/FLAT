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
from data_augments import *

PI = 3.14159265358979323846
raw_depth_new = 0
flg = False

dtype = tf.float32

def metric_valid(depth, gt, msk):
    # compute mean absolute error on places where msk = 1
    msk /= np.sum(msk)
    return np.sum(np.abs(depth - gt)*msk)

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

def testing_syn_motion(tests, array_dir, output_dir, tof_cam, tof_net):
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
        x_true = []
        z_gts = []
        for i in range(len(te_idx)):
            if np.mod(i,10)==0:
                scenes = select_objects()
            x_te,y_te,x_tr,z_gt = data_augment_th(scenes, array_dir, tof_cam)
            x.append(x_te)
            y.append(y_te)
            x_true.append(x_tr)
            z_gts.append(z_gt)
        x = np.stack(x,0)
        y = np.stack(y,0)
        x_true = np.stack(x_true, 0)
        z_gts = np.stack(z_gts, 0)

        # # chooose from the data
        # num = 1
        # idx = np.random.choice(x.shape[0],num,replace=False)
        # x = x[idx]
        # y = y[idx]

        # evaluate the model and print results
        eval_results = tof_net.evaluate(x=x,y=y)

        # predict data
        data = list(tof_net.predict(x=x))
        mid = 4
        for j in range(len(data)):
            ims_warped = []
            for k in range(9):
                msk = x[j,:,:,k+9]
                v = np.stack([data[j]['v'][:,:,k],data[j]['v'][:,:,k+9]],-1)
                ims = np.stack([x[j,:,:,k],x[j,:,:,mid]],-1)

                # visualize optical flow
                v_gt = np.stack([y[j,:,:,k],y[j,:,:,k+9]],-1)
                im_v_gt = viz_flow(v_gt[:,:,0],v_gt[:,:,1])
                im_v_pred = viz_flow(v[:,:,0],v[:,:,1])
                msk_exp = np.expand_dims(msk,-1)
                err = np.sum(np.abs(v - v_gt)*msk_exp)/np.sum(msk_exp)

                # expand the contour of v to solve the convex hull problem
                kl = np.ones([3,3])
                flg_int = signal.convolve2d(msk!=0, kl, mode='same')
                flg_edge = (flg_int!=0)-(msk!=0)

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
                v_xgt = y[j,:,:,1].flatten()
                v_ygt = y[j,:,:,0].flatten()
                xx_new = xx + v_x
                yy_new = yy + v_y
                # f = scipy.interpolate.interp2d(yy_new, xx_new, x[j,:,:,0], fill_value=0)
                pts = np.stack([yy_new,xx_new],-1)
                vals = ims[:,:,0].flatten()
                # vals[np.where(msk.flatten()==0)]=np.nan
                
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
                # vmin = vmin - (vmax - vmin)*3 # adjust the range to visualize the difference
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
                plt.title('Before align error: '+('%4f' % err_wo)+'.')
                plt.axis('off')
                fig.add_subplot(2,2,2)
                plt.imshow(im_v_gt)
                plt.axis('off')
                fig.add_subplot(2,2,3)
                plt.imshow(x_new1)
                plt.title('After align error: '+('%4f' % err_w)+'.')
                plt.axis('off')
                fig.add_subplot(2,2,4)
                plt.imshow(im_v_pred)
                plt.title('Mean error: '+('%.2f' % err)+' pixels.')
                plt.axis('off')
                            
                name = int(np.random.uniform()*1e10)
                plt.savefig(\
                    output_dir+str(name)+'.png',
                    bbox_inches='tight',
                    dpi = 2*512,
                )

            x_warped = np.expand_dims(np.stack(ims_warped,-1),0)
            x_or = x[j,:,:,0:9]
            msk = np.expand_dims(x[j,:,:,mid+9],-1)
            depth = np.zeros(msk.shape)
            y_or = np.expand_dims(np.concatenate([depth,msk,x_or],-1),0)
            x_or = np.expand_dims(x_or,0)
            x_gt = x_true[j:j+1,:,:,0:9]

            name = int(np.random.uniform()*1e10)
            fig=plt.figure()
            plt.title("Original image")
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
            plt.title("Warped image")
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
            x_gt = np.concatenate([np.zeros([1,20,512,9]),x_gt,np.zeros([1,20,512,9])],1)
            y_or = y_or.astype(np.float32)
            x_or = x_or.astype(np.float32)
            x_warped = x_warped.astype(np.float32)
            x_gt = x_gt.astype(np.float32)
            z_gt = z_gts[j,:,:,mid]
            z_gt = np.concatenate([np.zeros([20,512]),z_gt,np.zeros([20,512])],0)

            # remask the raw measurement
            msk = kinect_mask().astype(np.float32)
            x_or = [x_or[0,:,:,i]*msk*tof_cam.cam['map_max'] for i in range(x_or.shape[-1])]
            x_or = np.expand_dims(np.stack(x_or,-1),0)
            x_warped = [x_warped[0,:,:,i]*msk*tof_cam.cam['map_max'] for i in range(x_warped.shape[-1])]
            x_warped = np.expand_dims(np.stack(x_warped,-1),0)
            x_gt = [x_gt[0,:,:,i]*msk*tof_cam.cam['map_max'] for i in range(x_gt.shape[-1])]
            x_gt = np.expand_dims(np.stack(x_gt,-1),0)

            # compute depth
            eval_results = raw_depth_new.evaluate(x=x_or,y=y_or)
            depth_or = list(raw_depth_new.predict(x=x_or))[0]['depth']
            depth = list(raw_depth_new.predict(x=x_warped))[0]['depth']
            depth_gt = list(raw_depth_new.predict(x=x_gt))[0]['depth']
            

            # 
            vmin = z_gt[np.where(z_gt>1e-4)].min()
            vmax = z_gt.max()
            fig = plt.figure()
            ax = fig.add_subplot(2,4,1)
            plt.imshow(depth_or,vmin=vmin,vmax=vmax)
            msk = depth_or > 0.5
            err = np.sum(np.abs(depth_or - z_gt)*msk)/np.sum(msk)
            plt.title('Original raw, err: '+'%.4f'%err+'m')
            plt.axis('off')

            ax = fig.add_subplot(2,4,2)
            plt.imshow((depth_or - z_gt)*msk, vmin=-0.1,vmax=0.1)
            plt.axis('off')

            ax = fig.add_subplot(2,4,3)
            plt.imshow(depth,vmin=vmin,vmax=vmax)
            msk = depth > 0.5
            err = np.sum(np.abs(depth - z_gt)*msk)/np.sum(msk)
            plt.title('Corrected raw, err: '+'%.4f'%err+'m')
            plt.axis('off')

            ax = fig.add_subplot(2,4,4)
            plt.imshow((depth - z_gt)*msk, vmin=-0.1,vmax=0.1)
            plt.axis('off')

            ax = fig.add_subplot(2,4,5)
            plt.imshow(depth_gt,vmin=vmin,vmax=vmax)
            msk = depth_gt > 0.5
            err = np.sum(np.abs(depth_gt - z_gt)*msk)/np.sum(msk)
            plt.title('Ground truth raw, err: '+'%.4f'%err+'m')
            plt.axis('off')

            ax = fig.add_subplot(2,4,6)
            plt.imshow((depth_gt - z_gt)*msk, vmin=-0.1,vmax=0.1)
            plt.axis('off')

            ax = fig.add_subplot(2,4,7)
            plt.imshow(z_gt,vmin=vmin,vmax=vmax)
            plt.title('Ground truth raw')
            plt.axis('off')
            plt.colorbar()

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
        x_true = []
        z_gts = []
        for i in range(len(te_idx)):
            # if np.mod(i,10)==0:
            #     scenes = select_objects()
            # x_te,y_te,x_tr,z_gt = data_augment_th(scenes, array_dir, tof_cam)
            x_te,y_te,x_tr,z_gt = data_augment_real(tests[te_idx[i]], array_dir, tof_cam)
            x.append(x_te)
            y.append(y_te)
            x_true.append(x_tr)
            z_gts.append(z_gt)
        x = np.stack(x,0)
        y = np.stack(y,0)
        x_true = np.stack(x_true, 0)
        z_gts = np.stack(z_gts, 0)

        # # chooose from the data
        # num = 1
        # idx = np.random.choice(x.shape[0],num,replace=False)
        # x = x[idx]
        # y = y[idx]

        # evaluate the model and print results
        eval_results = tof_net.evaluate(x=x,y=y)

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
                v_gt = np.stack([y[j,:,:,k],y[j,:,:,k+9]],-1)
                im_v_gt = viz_flow(v_gt[:,:,0],v_gt[:,:,1])
                im_v_pred = viz_flow(v[:,:,0],v[:,:,1])
                msk_exp = np.expand_dims(msk,-1)
                err = np.sum(np.abs(v - v_gt)*msk_exp)/np.sum(msk_exp)

                # expand the contour of v to solve the convex hull problem
                kl = np.ones([3,3])
                flg_int = signal.convolve2d(msk!=0, kl, mode='same')
                flg_edge = (flg_int!=0)-(msk!=0)

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
                v_xgt = y[j,:,:,1].flatten()
                v_ygt = y[j,:,:,0].flatten()
                xx_new = xx + v_x
                yy_new = yy + v_y
                # f = scipy.interpolate.interp2d(yy_new, xx_new, x[j,:,:,0], fill_value=0)
                pts = np.stack([yy_new,xx_new],-1)
                vals = ims[:,:,0].flatten()
                # vals[np.where(msk.flatten()==0)]=np.nan
                
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
                # vmin = vmin - (vmax - vmin)*3 # adjust the range to visualize the difference
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
                plt.title('Before align error: '+('%4f' % err_wo)+'.')
                fig.add_subplot(2,2,2)
                plt.imshow(im_v_gt)
                fig.add_subplot(2,2,3)
                plt.imshow(x_new1)
                plt.title('After align error: '+('%4f' % err_w)+'.')
                fig.add_subplot(2,2,4)
                plt.imshow(im_v_pred)
                plt.title('Mean error: '+('%.2f' % err)+' pixels.')
                            
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
            x_gt = x_true[j:j+1,:,:,0:9]

            name = int(np.random.uniform()*1e10)
            fig=plt.figure()
            plt.title("Original image")
            for i in range(9):
                ax=fig.add_subplot(3,3,i+1)
                plt.imshow(x_or[0,:,:,i])
            plt.savefig(\
                output_dir+str(name)+'.png',
                bbox_inches='tight',
                dpi = 2*512,
            )

            name = int(np.random.uniform()*1e10)
            fig=plt.figure()
            plt.title("Warped image")
            for i in range(9):
                ax=fig.add_subplot(3,3,i+1)
                plt.imshow(x_warped[0,:,:,i])
            plt.savefig(\
                output_dir+str(name)+'.png',
                bbox_inches='tight',
                dpi = 2*512,
            )

            # add shape
            y_or = np.concatenate([np.zeros([1,20,512,11]),y_or,np.zeros([1,20,512,11])],1)
            x_or = np.concatenate([np.zeros([1,20,512,9]),x_or,np.zeros([1,20,512,9])],1)
            x_warped = np.concatenate([np.zeros([1,20,512,9]),x_warped,np.zeros([1,20,512,9])],1)
            x_gt = np.concatenate([np.zeros([1,20,512,9]),x_gt,np.zeros([1,20,512,9])],1)
            y_or = y_or.astype(np.float32)
            x_or = x_or.astype(np.float32)
            x_warped = x_warped.astype(np.float32)
            x_gt = x_gt.astype(np.float32)
            z_gt = z_gts[j,:,:]
            z_gt = np.concatenate([np.zeros([20,512]),z_gt,np.zeros([20,512])],0)

            # remask the raw measurement
            msk = kinect_mask().astype(np.float32)
            x_or = [x_or[0,:,:,i]*msk*tof_cam.cam['map_max'] for i in range(x_or.shape[-1])]
            x_or = np.expand_dims(np.stack(x_or,-1),0)
            x_warped = [x_warped[0,:,:,i]*msk*tof_cam.cam['map_max'] for i in range(x_warped.shape[-1])]
            x_warped = np.expand_dims(np.stack(x_warped,-1),0)
            x_gt = [x_gt[0,:,:,i]*msk*tof_cam.cam['map_max'] for i in range(x_gt.shape[-1])]
            x_gt = np.expand_dims(np.stack(x_gt,-1),0)

            # compute depth
            eval_results = raw_depth_new.evaluate(x=x_or,y=y_or)
            depth_or = list(raw_depth_new.predict(x=x_or))[0]['depth']
            depth = list(raw_depth_new.predict(x=x_warped))[0]['depth']
            depth_gt = list(raw_depth_new.predict(x=x_gt))[0]['depth']
            

            # 
            vmin = z_gt[np.where(z_gt>1e-4)].min()
            vmax = z_gt.max()
            fig = plt.figure()
            ax = fig.add_subplot(2,4,1)
            plt.imshow(depth_or,vmin=vmin,vmax=vmax)
            msk = depth_or > 0.5
            err = np.sum(np.abs(depth_or - z_gt)*msk)/np.sum(msk)
            plt.title('Original raw, err: '+'%.4f'%err+'m')

            ax = fig.add_subplot(2,4,2)
            plt.imshow((depth_or - z_gt)*msk, vmin=-0.1,vmax=0.1)

            ax = fig.add_subplot(2,4,3)
            plt.imshow(depth,vmin=vmin,vmax=vmax)
            msk = depth > 0.5
            err = np.sum(np.abs(depth - z_gt)*msk)/np.sum(msk)
            plt.title('Corrected raw, err: '+'%.4f'%err+'m')

            ax = fig.add_subplot(2,4,4)
            plt.imshow((depth - z_gt)*msk, vmin=-0.1,vmax=0.1)

            ax = fig.add_subplot(2,4,5)
            plt.imshow(depth_gt,vmin=vmin,vmax=vmax)
            msk = depth_gt > 0.5
            err = np.sum(np.abs(depth_gt - z_gt)*msk)/np.sum(msk)
            plt.title('Ground truth raw, err: '+'%.4f'%err+'m')

            ax = fig.add_subplot(2,4,6)
            plt.imshow((depth_gt - z_gt)*msk, vmin=-0.1,vmax=0.1)

            ax = fig.add_subplot(2,4,7)
            plt.imshow(z_gt,vmin=vmin,vmax=vmax)
            plt.title('Ground truth raw')
            plt.colorbar()

            ax = fig.add_subplot(2,4,8)
            plt.imshow((z_gt - z_gt)*msk, vmin=-0.1,vmax=0.1)
            plt.colorbar()         

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
    f = open('../FLAT/kinect/list/motion_real.txt','r')
    message = f.read()
    files = message.split('\n')
    tests = files[0:-1]
    tests = [data_dir+test for test in tests]

    # # initialize the camera model
    # import training
    # tof_cam = kinect_real_tf()
    # training.tof_cam = tof_cam

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
    output_dir = './results/kinect/'    
    folder_name = file_name    
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output_dir = output_dir + folder_name + '/'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    testing_syn_motion(tests, array_dir, output_dir, tof_cam, tof_net)
    # testing_real_motion(tests, array_dir, output_dir, tof_cam, tof_net)
