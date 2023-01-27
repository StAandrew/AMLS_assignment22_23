# Description of the project

[Project](https://github.com/StAandrew/AMLS_assignment22_23)

This project provides possible solutions to two binary classification tasks for gender (A1) and smile detection (A2) and two multi-categorical classification tasks concerning face-shape (B1) and eye-colour (B2) recognition. 

|                                       |                                   Task A1                                                  |                                            Task A2                                      |                                        Task B1                                                                                   |                                     Task B2                                    |
| ------------------------------------- | :----------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------: |
| Dataset                               |                    CelebA                                                                  |                                                 CelebA                                  |           Cartoon Set                                                                                                            |                                     Cartoon Set                                |
| Number of samples                     |                 5.000 images                                                               |                                              5.000 images                               |          10.000 images                                                                                                           |                                    10.000 images                               |
| Original size of each image           |                  178x218x3                                                                 |                                               178x218x3                                 |            500x500x3                                                                                                             |                                      500x500x3                                 |
| Target                                |                      Male or female                                                        |                     Smiling or not smiling                                              |             Face shape type 0-4                                                                                                  |                                 Eye colour 0-4                                 |
| Pre-processing                        | Jaw features are extracted by using a pre-trained HOG and Linear SVM face detector by Dlib |   Smiles are extracted by using a pre-trained HOG and Linear SVM face detector by Dlib  | Images are cropped around jaw area, converted into grayscale, (optional) masked, (optional) images with full beards are removed) | Images are cropped around left eye, images containing black sunglasses removed |
| Input example shape                   |                    16x2                                                                    |                                                 20x2                                    |            130x190                                                                                                               |                                      27x120x3                                  |
| Model                                 |                     SVM                                                                    |                                                  SVM                                    |               CNN                                                                                                                |                                        CNN                                     |
| Batch size                            |                      NA                                                                    |                                                  N/A                                    |                16                                                                                                                |                                          16                                    |
| Epoch count                           |                      NA                                                                    |                                                  N/A                                    |                10                                                                                                                |                                         10                                     |


# Instructions

## Setup
**Note:** CSH shell specific instructions. Modify accordingly for BASH or ZSH shell.
1. Install python 3.10 or higher  
2. Install Pypi 
3. Clone the project from [GitHub](https://github.com/StAandrew/AMLS_assignment22_23).
4. Create a virtual environment.
```csh
   python -m venv venv
   ```
   Activate the newly created virtual environment.
```csh
   source venv/bin/activate.csh
   ```
5. Install packages from requirements.txt file. 
```python
   pip install -r requirements.txt
   ```
6. Save a dlib pre-trained 68 facial landmarks model to project root directory ([at this link]([https://github.com/italojs/facial-landmarks-recognition/blob/master/shape_predictor_68_face_landmarks.dat])(https://github.com/italojs/facial-landmarks-recognition/blob/master/shape_predictor_68_face_landmarks.dat).
7. Download datasets and extract zip folders into Datasets folder
8. Resolve issues with Tensorflow. On the particular machine the following commands resolved the issues.
```csh
   setenv XLA_FLAGS --xla_gpu_cuda_data_dir=/apps/cuda/cuda-11.2.0
   ```
```csh
   cd venv/lib/python3.10/site-packages/tensorrt
   ```
```csh
   ln -s libnvinfer.so.8 libnvinfer.so.7
   ```
```csh
   ln -s libnvinfer_plugin.so.8 libnvinfer_plugin.so.6
   ```
```csh
   setenv LD_LIBRARY_PATH $LD_LIBRARY_PATH\\:/apps/cuda/cuda-11.2.0/nvvm/libdevice
   ```
   Get a current directory and copy it:
```csh
   pwd
   ```
   Paste the current directory instead of \<CURRENT_DIR\> before running the following command:
```csh
   setenv LD_LIBRARY_PATH $LD_LIBRARY_PATH\\:<CURRENT_DIR>/venv/lib/python3.10/site-packages/tensorrt
   ```
   
## Run the code

Now you can run the project.
```python
python -m main
```

## Datasets

The datasets can be downloaded via following links:  
[dataset_AMLS_22-23](https://bit.ly/dataset_AMLS_22-23) ([https://bit.ly/dataset_AMLS_22-23](https://bit.ly/dataset_AMLS_22-23))  
[dataset_AMLS_22-23_test](https://bit.ly/dataset_AMLS_22-23_test) ([https://bit.ly/dataset_AMLS_22-23_test](https://bit.ly/dataset_AMLS_22-23_test))  

### CelebA dataset
CelebFaces Attributes Dataset (CelebA), a celebrity image dataset (S. Yang, P. Luo, C. C. Loy, and X. Tang, "From facial parts responses to face detection: A Deep Learning Approach", in IEEE International Conference on Computer Vision (ICCV), 2015) [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)  
The dataset provided contains 5000 celebrity images. Each one of them is associated to two labels that describe the celebrity gender and whether they are smiling. The table below depicts how the two categories for each label are divided.

| Class | Gender | Smiling |
| ----- | ------ | ------- |
| -1    | 2500   | 2500    |
| 1     | 2500   | 2500    |

**Note:** In the code -1 is mapped to 0, in order to use uint8 numpy data format.

### Cartoon_set dataset
Cartoon Set, an image dataset of random cartoons/avatars (source: https://google.github.io/cartoonset/). [Cartoon_set](https://google.github.io/cartoonset/download.html)  
The dataset is made up of 10000 avatar images. Each character is defined by 18 components including 10 artwork attributes, 4 proportion attributes and 4 colour attributes. The following table summarizes how the examples are distributed amongst the five possibilities for both: eye color and face shape.

| Class | Eye color | Face shape | Eye color (no black sunglasses) |
| ----- | --------- | ---------- | ------------------------------- |
| 0     | 2004      | 2000       | 1643                            |
| 1     | 2018      | 2000       | 1654                            |
| 2     | 1969      | 2000       | 1593                            |
| 3     | 1992      | 2000       | 1621                            |
| 4     | 2017      | 2000       | 1635                            |


## Models



**CNN model for Task B1**

| Layer (type)       | Output shape    | Parameters  |
| ------------------ | --------------- | ----------- |
| Input              | ( , 130, 190, 1) | 0          |
| Convolutional_2D   | ( , 130, 190, 16)| 160        |
| MaxPooling_2D      | ( , 65, 95, 16)  | 0          |
| Convolutional_2D   | ( , 65, 95, 32)  | 4640       |
| MaxPooling_2D      | ( , 32, 47, 32)  | 0          |
| Convolutional_2D   | ( , 32, 47, 64)  | 18496      |
| MaxPooling_2D      | ( , 16, 23, 64)  | 0          |
| Flatten            | ( , 23552)       | 0          |
| Dense              | ( , 1024)        | 24118272   |
| Dropout            | ( , 1024)        | 0          |
| Dense              | ( , 5)           | 5125       |
| Total params       |                  | 24146693   |

**CNN model for Task B2**

| Layer (type)       | Output shape      | Parameters |
| ------------------ | ----------------- | ---------- |
| Input              | ( , 27, 120, 3)   | 0          |
| Convolutional_2D   | ( , 27, 120, 16)  | 448        |
| MaxPooling_2D      | ( , 13, 60, 16)   | 0          |
| Convolutional_2D   | ( , 13, 60, 32)   | 4640       |
| MaxPooling_2D      | ( , 6, 30, 32)    | 0          |
| Convolutional_2D   | ( , 6, 30, 64)    | 18496      |
| MaxPooling_2D      | ( , 3, 15, 64)    | 0          |
| Flatten            | ( , 2880)         | 0          |
| Dense              | ( , 1024)         | 2950144    |
| Dropout            | ( , 1024)         | 0          |
| Dense              | ( , 5)            | 5125       |
| Total params       |                   | 2978565    |
