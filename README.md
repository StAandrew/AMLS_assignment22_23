# AMLS_assignment22_23  
  
Two datasets which are designed specifically for this assignment and contains pre-processed subsets from the following datasets:
1. CelebFaces Attributes Dataset (CelebA), a celebrity image dataset (S. Yang, P. Luo, C. C. Loy, and X. Tang, "From facial parts responses to face detection: A Deep Learning Approach", in IEEE International Conference on Computer Vision (ICCV), 2015)  
2. Cartoon Set, an image dataset of random cartoons/avatars (source: https://google.github.io/cartoonset/).  
  
The datasets that need to be imported are:  
1. celeba: A sub-set of CelebA dataset. This dataset contains 5000 images. It is going to be used for A1 and A2.  
2. cartoon_set: A subset of Cartoon Set. This dataset contains 10000 images. It is going to be used for B1 and B2.  
  
The datasets can be downloaded via following link:  
https://bit.ly/dataset_AMLS_22-23  
A separate test set:  
https://bit.ly/dataset_AMLS_22-23_test  
  
  
  
For training on UCL GPU servers use the following:

Install python3.10 for the current user (since there are no sudo privileges) 

python3 -m venv venv

pip3 install -r requirements.txt

pwd

setenv XLA_FLAGS --xla_gpu_cuda_data_dir=/apps/cuda/cuda-11.2.0

setenv CUDA_DIR /apps/cuda/cuda-11.2.0

setenv LD_LIBRARY_PATH $LD_LIBRARY_PATH\:<last command output>/venv/lib/python3.10/site-packages/tensorrt

setenv LD_LIBRARY_PATH $LD_LIBRARY_PATH\:/apps/cuda/cuda-11.2.0/nvvm/libdevice
  
cd /venv/lib/python3.10/site-packages/tensorrt

ln -s libnvinfer.so.8 libnvinfer.so.7

ln -s libnvinfer_plugin.so.8 libnvinfer_plugin.so.6

To run a module: python3 -m A1.a1
To run the entire project: python3 -m main
