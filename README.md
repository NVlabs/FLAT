# Flexible Large Augmentable Time-of-flight (FLAT) Dataset

This repository provides access to the FLAT dataset and some proposed baseline methods for time-of-flight (ToF) artifacts removal. The FLAT dataset includes synthetic measurements of several scenes that allow generaring raw measurements for different ToF cameras, such as Kinect 2, including motion blur, multi-path interference (MPI), and shot noise. This repository also provides a joint physics-and-learning-based method that reduces these artifacts. For details of the dataset and the methods, please refer to the [project page](https://research.nvidia.com/publication/2018-09_Tackling-3D-ToF). If you use this code in your work, please kindly cite the following paper:

```
Qi Guo, Iuri Frosio, Orazio Gallo, Todd Zickler and Jan Kautz. 
"Tackling 3D ToF Artifacts Through Learning and the FLAT Dataset." 
In Proc. European Conference on Computer Vision (ECCV), 2018.
```

## Prerequisites
The provided methods and interface are implemented in Python 3.5. Necessary packages include [TensorFlow 1.10.0](https://www.tensorflow.org/install/) (this requires CUDA 9.0 for GPU usage), [OpenCV 3.1.0](https://docs.opencv.org/3.1.0/), and googledrivedownloader (can be installed with `pip install googledrivedownloader --user`).

## Usage
0. Make sure your Python environment satisfies the prerequisites.
1. Clone the repository (git clone https://github.com/guoqi1123/FLAT_pub.git). 
2. Run `python init.py`, which will download a small fraction of synthetic raw measurements of Kinect 2, that is necessary for testing, also it will automatically download miscellaneous data files. If you would like to download all synthetic raw measurements of Kinect 2, run `python init.py -n all`. If you would like to download synthetic raw measurements of hardware described in DeepToF [1] or Phasor [2], run `python init.py -c HARDWARE_NAME`. If you would like to download the transient rendering files (generated using the transient rendering framework by Jarabo et al. [3]) and produce raw measurements of your own camera, run  `python init.py -c trans_render`. **WARNING**: the total size of transient rendering files is about 800GB.
3. Run `python testing_NETWORK_NAME.py -n 1` in `./pipe/`, where `NETWORK_NAME` is the method name you want to test, e.g., `MOM_MRM_LF2`; this will process the first image of the dataset only. To process all images, use '-n -1' or no additional arguments. Visualization results will be in `./pipe/results`.

## Organization of the folders
The FLAT dataset is organized in the following way.

./FLAT

The folder `./FLAT/trans_render/` contains transient rendering images. Among them, the static scenes are in `./FLAT/trans_render/static/`, the scenes with motion are in `./FLAT/trans_render/dyn/`. For now, transient rendering of 121 static scenes, and 56 dynamic scenes is online. WARNING: the total size is 576GB. Each of the `.pickle` file in the folder contains the transient rendering of a scene. One can load the data using the following code:

```
with open(FILENAME,'rb') as f:
	data = pickle.load(f)

	# copy the variables
	cam = data['cam'] # details of the camera setting
	scene = data['scene'] # details of the scene
	depth_true = data['depth_true'] # true depth map
	
	# the transient rendering is saved in a sparse matrix
	# prop_idx saves the index, prop_s saves the value at each idx
	prop_idx = data['prop_idx'] 
	prop_s = data['prop_s']
```

The code of generating raw measurements of a certain platform from the transient rendering is in `./sim/`, e.g., `./sim/kinect_prepare.py` generates kinect raw measures.

The folder `./FLAT/kinect/` contains simulated kinect 2 raw measurements of 1929 scenes. For each scene, we create raw measurements in different settings, stored in different subfolders.


 `./FLAT/kinect/full` contains raw measurements closely resembles kinect 2, `./FLAT/kinect/reflection` is with 
The folder ...

## Reconstructing a scene using pretrained networks
To reconstruct a scene using a pretrained network... The input is the transient rendering... Data augmentation can be done in this way... The result will be in...

## Reference
[1] Marco, J., Hernandez, Q., Mu&#x00F1;oz, A., Dong, Y., Jarabo, A., Kim, M.H., Tong, X., Gutierrez, D.: DeepToF: Off-the-shelf real-time correction of multipath interference in time-of-flight imaging. In: ACM Transactions on Graphics (SIGGRAPH ASIA). (2017)

[2] Gupta, M., Nayar, S.K., Hullin, M.B., Martin, J.: Phasor imaging: A generalization of correlation-based time-of-flight imaging. ACM Transactions on Graphics. (2015)

[3] Jarabo, A., Marco, J., Mu&#x00F1;oz, A., Buisan, R., Jarosz, W., Gutierrez, D.: A framework for transient rendering. In: ACM Transactions on Graphics (SIGGRAPH ASIA). (2014) 
