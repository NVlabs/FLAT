# Flexible Large Augmentable Time-of-flight (FLAT) Dataset

This repository provides access to the FLAT dataset and some proposed baseline methods for time-of-flight (ToF) artifacts removal. The FLAT dataset includes synthetic measurements of several scenes that allow generaring raw measurements for different ToF cameras, such as Kinect 2, including motion blur, multi-path interference (MPI), and shot noise. This repository also provides a joint physics-and-learning-based method that reduces these artifacts. For details of the dataset and the methods, please refer to the [project page](https://research.nvidia.com/publication/2018-09_Tackling-3D-ToF). If you use this code in your work, please kindly cite the following paper:

```
Qi Guo, Iuri Frosio, Orazio Gallo, Todd Zickler and Jan Kautz. 
"Tackling 3D ToF Artifacts Through Learning and the FLAT Dataset." 
In Proc. European Conference on Computer Vision (ECCV), 2018.
```

## Prerequisites
The provided methods and interface are implemented in Python 3.5. Necessary packages include [TensorFlow 1.4.0](https://www.tensorflow.org/install/) (this requirese CUDA 9.0 for GPU usage), [OpenCV 3.1.0](https://docs.opencv.org/3.1.0/), and googledrivedownloader (can be installed with `pip install googledrivedownloader --user`).

## Usage
0. Make sure your Python environment satisfies the prerequisites.
1. Clone the repository (git clone https://github.com/guoqi1123/FLAT_pub.git). 
2. Run `python init.py`, which will download a small fraction of synthetic raw measurements of Kinect 2, that is necessary for testing, also it will automatically download miscellaneous data files. If you would like to download all synthetic raw measurements of Kinect 2, run `python init.py -n all`. If you would like to download synthetic raw measurements of hardware described in DeepToF [1] or Phasor [2], run `python init.py -c HARDWARE_NAME`. If you would like to download the transient rendering files (generated using the transient rendering framework by Jarabo et al. [3]) and produce raw measurements of your own camera, run  `python init.py -c trans_render`. **WARNING**: the total size of transient rendering files is about 800GB.
3. Run `python testing_NETWORK_NAME.py` in `./pipe/`, where `NETWORK_NAME` is the method name you want to test, e.g., `MOM_MRM_LF2`. Visualization results will be in `./pipe/results`.

## Reference
[1] Marco, J., Hernandez, Q., Mu&#x00F1;oz, A., Dong, Y., Jarabo, A., Kim, M.H., Tong, X., Gutierrez, D.: DeepToF: Off-the-shelf real-time correction of multipath interference in time-of-flight imaging. In: ACM Transactions on Graphics (SIGGRAPH ASIA). (2017)

[2] Gupta, M., Nayar, S.K., Hullin, M.B., Martin, J.: Phasor imaging: A generalization of correlation-based time-of-flight imaging. ACM Transactions on Graphics. (2015)

[3] Jarabo, A., Marco, J., Mu&#x00F1;oz, A., Buisan, R., Jarosz, W., Gutierrez, D.: A framework for transient rendering. In: ACM Transactions on Graphics (SIGGRAPH ASIA). (2014) 
