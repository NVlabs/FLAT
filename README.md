# Flexible Large Augmentable Time-of-flight (FLAT) Dataset

This repository provides access to the FLAT dataset and some proposed baseline methods for time-of-flight (ToF) artifacts removal. The FLAT dataset includes synthetic measurements of several scenes that allow generaring raw measurements for different ToF cameras, such as Kinect 2, including motion blur, multi-path interference (MPI), and shot noise. This repository also provides a joint physics-and-learning-based method that reduces these artifacts. For details of the dataset and the methods, please refer to the [project page](https://research.nvidia.com/publication/2018-09_Tackling-3D-ToF). If you use this code in your work, please kindly cite the following paper:

```
@InProceedings{Guo_2018_ECCV,
author = {Guo, Qi and Frosio, Iuri and Gallo, Orazio and Zickler, Todd and Kautz, Jan},
title = {Tackling 3D ToF Artifacts Through Learning and the FLAT Dataset},
booktitle = {The European Conference on Computer Vision (ECCV)},
month = {September},
year = {2018}
}
```

## Installing FLAT locally

### Prerequisites
The provided methods and interface are implemented in Python 3.5. Necessary packages include [TensorFlow 1.9.0](https://www.tensorflow.org/install/) (this requires CUDA 9.0 for GPU usage), [OpenCV 3.1.0](https://docs.opencv.org/3.1.0/), and googledrivedownloader (can be installed with `pip install googledrivedownloader --user`).

### Downloading the dataset
0. Make sure your Python environment satisfies the prerequisites.
1. Clone the repository (git clone https://github.com/NVlabs/FLAT.git). 
2. Run `python init.py`, which will download a small fraction of synthetic raw measurements of Kinect 2, that is necessary for testing, also it will automatically download miscellaneous data files. If you would like to download all synthetic raw measurements, run `python init.py -n all`. If you would like to download synthetic raw measurements of hardware described in DeepToF [1], Phasor [2] or the Kinect 2, run `python init.py -c HARDWARENAME`, where `HARDWARENAME` can be `deeptof`, `phasor` or `kinect`. If you would like to download the transient rendering files (generated using the transient rendering framework by Jarabo et al. [3]) and produce raw measurements of your own camera, run  `python init.py -c trans_render`. **WARNING**: the total size of transient rendering files is about 576GB.

## Docker installation

If you prefer using docker, you need [docker](https://www.docker.com/get-started) and [nvidia-docker 2.0](https://github.com/nvidia/nvidia-docker/wiki/Installation-(version-2.0)) on your machine.
All the prerequisites, code, and the FLAT dataset can be obtained by building a docker image using the following docker file - just remember to change your github username and password in the docker file:

```
FROM nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04
	
RUN apt-get update
RUN apt-get -y install python3
RUN apt-get -y install python3-pip
RUN pip3 install tensorflow-gpu==1.9.0
RUN pip3 install opencv-python
RUN apt-get -y install libglib2.0-0 libsm6 libxext6 libgtk2.0-dev
RUN pip3 install requests
RUN pip3 install googledrivedownloader 

RUN apt-get update
RUN apt-get -y install git --fix-missing
RUN git clone https://<YOUR_GITHUB_UN>:<YOUR_GITHUB_pw>@github.com/guoqi1123/FLAT_pub.git

WORKDIR FLAT_pub
RUN pip3 install imageio matplotlib scipy joblib
RUN python3 init.py

```

Copy and paste the docker file and save it as "Dockerfile" in your preferred folder. 
To create the docker image, run:

```
sudo nvidia-docker build -t FLAT .
```

To run it after its creation:

```
sudo docker run -it --runtime=nvidia --rm FLAT
```

## Organization and access of the FLAT dataset
 The FLAT dataset is organized in the following way:

```
FLAT
├───deeptof		# simulation of DeepToF
│   ├───full			# raw measurements
│   └───list			# lists of files to use for each task, automatically generated from ./sim/deeptof_prepare.py
├───kinect		# simulation of kinect 2
│   ├───full			# raw measurements
│   ├───noise			# raw measurements without multi path interference (MPI), with noise
│   ├───ideal			# raw measurements without MPI and noise
│   ├───reflection		# raw measurements with MPI, and without noise
│   ├───gt			# true depth
│   ├───list			# lists of files to use for each task, automatically generated from ./sim/kinect_prepare.py
│   │   ├───all				# visualization of all scenes 
│   │   ├───motion_background		# visualization of scenes for a certain task
│   │   ├───motion_foreground		# visualization of scenes for a certain task
│   │   ├───motion_real			# visualization of scenes for a certain task
│   │   ├───shot_noise_test		# visualization of scenes for a certain task
│   │   ├───test			# visualization of scenes for a certain task
│   │   ├───test_dyn			# visualization of scenes for a certain task
│   │   ├───train			# visualization of scenes for a certain task
│   │   └───val				# visualization of scenes for a certain task
│   └───msk			# the mask of background
├───phasor		# simulation of Phasor
│   ├───full			# raw measurements
│   │   ├───FILE_ID		
│   └───list			# lists of files to use for each task, automatically generated from ./sim/phasor_prepare.py
└───trans_render	# transient rendering files
    ├───dyn			# dynamic scenes
    └───static			# static scenes
```

The folder `./FLAT/trans_render/` contains transient rendering images. Among them, the static scenes are in `./FLAT/trans_render/static/`, the scenes with motion are in `./FLAT/trans_render/dyn/`. For now, transient rendering of 121 static scenes, and 56 dynamic scenes are available. WARNING: the total size is 576GB. Each of the `.pickle` file in the folder contains the transient rendering of a scene. One can load the data using the following code:

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

Raw measurements of new scenes can be created by running `./sim/HARDWARENAME_prepare.py`, where `HARDWARENAME` can be `deeptof`, `phasor`, or `kinect`, and new transient rendering pickle files are saved in `./FLAT/trans_render/`. The raw kinect measurements can be loaded using the following code:

```
with open(filename,'rb') as f:
	meas=np.fromfile(f, dtype=np.int32)
meas = np.reshape(meas,(424,512,9)).astype(np.float32)
```
The raw measurements of deeptof and phasor can be accessed following the original code ([deeptof](http://webdiis.unizar.es/~juliom/pubs/2017SIGA-DeepToF/), [phasor](http://www.cs.columbia.edu/CAVE/projects/phasor_imaging/)).

## Reconstructing a scene using pretrained networks
Run `python testing_NETWORK_NAME.py -n 1` in `./pipe/`, where `NETWORK_NAME` is the method name you want to test, e.g., `MOM_MRM_LF2`; this will process the first image of the dataset only. To process all images, use '-n -1' or no additional arguments.
Results will be saved as images in `./pipe/results`. To modify how the results are saved, one can refer to the `testing` function in each file.

## Reference
[1] Marco, J., Hernandez, Q., Mu&#x00F1;oz, A., Dong, Y., Jarabo, A., Kim, M.H., Tong, X., Gutierrez, D.: DeepToF: Off-the-shelf real-time correction of multipath interference in time-of-flight imaging. In: ACM Transactions on Graphics (SIGGRAPH ASIA). (2017)

[2] Gupta, M., Nayar, S.K., Hullin, M.B., Martin, J.: Phasor imaging: A generalization of correlation-based time-of-flight imaging. ACM Transactions on Graphics. (2015)

[3] Jarabo, A., Marco, J., Mu&#x00F1;oz, A., Buisan, R., Jarosz, W., Gutierrez, D.: A framework for transient rendering. In: ACM Transactions on Graphics (SIGGRAPH ASIA). (2014) 
