# liteflownet-tf2
LiteFlowNet inference realization with TensorFlow 2.

Inspired by pytorch reimplenetation: https://github.com/sniklaus/pytorch-liteflownet

This is my reimplementation of LiteFlowNet [1] using Tensorflow 2.0 with build-in correlation layer. 
There are some messy code with using of tf2 package and it compatibility mode with tf 1.x.
Also, please cite the paper accordingly and make sure to adhere to the licensing terms of the authors.
Should you be making use of this particular implementation, please acknowledge it appropriately [3].

Original Caffe version: https://github.com/twhui/LiteFlowNet

## Setup
The correlation layer is using tensorflow_addons package which required tf2+ version. These repo tested with packages installed via pip:

```
pip install --user tensorflow==2.0.0
pip install --user tensorflow_addons==0.6.0
```

These weights converted from PyTorch version [2] via script convert_pytorch2tf.py

I convert only default model. You can convert it as youself. Or you can download already converted model from the link: \
https://drive.google.com/open?id=1apeRotQKMsFji8MKKNzcx-QO4udkJcdJ.

```
cd liteflownet-tf2

# download any of that models
wget --timestamping http://content.sniklaus.com/github/pytorch-liteflownet/network-default.pytorch
wget --timestamping http://content.sniklaus.com/github/pytorch-liteflownet/network-kitti.pytorch
wget --timestamping http://content.sniklaus.com/github/pytorch-liteflownet/network-sintel.pytorch

# convert default model
python convert_pytorch2tf.py --input_model=./network-default.pytorch --output_model=./model
```
## How to run
To run it on your own pair of images, use the following command.

```
python eval.py --img1=./images/first.png --img2=./images/second.png --flow=./out.flow --display_flow=True
```
Results differ a little. I think it depends on a bit different feature warping then in original work.
<p align="center"><img src="images/compare.gif?raw=true" alt="Comparison"></p>

## References
```
[1]  @inproceedings{Hui_CVPR_2018,
         author = {Tak-Wai Hui and Xiaoou Tang and Chen Change Loy},
         title = {{LiteFlowNet}: A Lightweight Convolutional Neural Network for Optical Flow Estimation},
         booktitle = {IEEE Conference on Computer Vision and Pattern Recognition},
         year = {2018}
     }
```

```
[2]  @misc{pytorch-liteflownet,
         author = {Simon Niklaus},
         title = {A Reimplementation of {LiteFlowNet} Using {PyTorch}},
         year = {2019},
         howpublished = {\url{https://github.com/sniklaus/pytorch-liteflownet}}
    }
```
```
[2]  @misc{liteflownet-tf2,
         author = {Vladimir Mikhelev},
         title = {{LiteFlowNet} inference realization with {Tensorflow 2}},
         year = {2020},
         howpublished = {\url{https://github.com/keeper121/liteflownet-tf2}}
    }

```
