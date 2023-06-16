# pytorch-video-recognition

 
## Introduction
This repo contains several models for video action recognition,
including C3D, R2Plus1D, R3D, inplemented using PyTorch .
Currently, we train these models on Bullying10K datasets.
 
 
## Installation
The code was tested with Anaconda and Python 3.9. After installing the Anaconda environment:

 
1. Install dependencies:

    For PyTorch dependency, see [pytorch.org](https://pytorch.org/) for more details.

    For custom dependencies:
    ```Shell
    conda install opencv
    pip install tqdm scikit-learn tensorboardX
    ```
 

3. Configure your dataset and pretrained model path.  

5. You can choose different models and datasets.

    To train the model, please do:
    ```Shell
    python train.py
    ```

## Datasets:

We used the dataset: Bullying10K.

## Source:

It repo is modified from https://github.com/jfzhang95/pytorch-video-recognition
