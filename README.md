# Flexible Large Augmentable Time-of-flight (FLAT) Dataset

This repository accompanies provides access to the FLAT dataset and some proposed baseline methods on time-of-flight (ToF) artifacts removal. The FLAT dataset provides synthetic raw measurements of different ToF cameras, such as Kinect 2, that simulates motion blur, multi-path interference (MPI), and shot noise. It also provides a joint physics-and-learning-based method that reduces these artifacts. For details of the dataset and the methods, please refer to the [project page](https://research.nvidia.com/publication/2018-09_Tackling-3D-ToF). If you use this code in your work, please kindly cite the following paper:

```
Qi Guo, Iuri Frosio, Orazio Gallo, Todd Zickler and Jan Kautz. 
"Tackling 3D ToF Artifacts Through Learning and the FLAT Dataset." 
In Proc. European Conference on Computer Vision (ECCV), 2018.
```

## Prerequisites
The provided methods and interface are implemented in Python 3.5. Necessary packages include [TensorFlow 1.4.0](https://www.tensorflow.org/install/) and [OpenCV 3.1.0](https://docs.opencv.org/3.1.0/).

## Usage
Clone 
