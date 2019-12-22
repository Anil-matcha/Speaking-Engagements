# Introduction to Deep Learning and Computer Vision

Learning paths today

1. Classical Machine Learning
2. Computer Vision
3. NLP
4. Reinforcement Learning



![](https://tkwsibf.edu.in/wp-content/uploads/2016/05/choice-confuse-1-800x285.png)



- There is no prerequisite between Deep Learning and Machine Learning

- You don't need to be good at maths to get started

- Pick up a broad field and break into topics which you need to conquer

- Be consistent and see improvements

    

# Goal

Set a goal for the new year and conquer it with little milestones

Project your work outside and let people recognize you

Collaborate with people and fast-track your progress



I had a goal to be good at deep learning at the beginning of the year and was able to complete few of the things I had in mind



# Deep Learning

Deep Learning doesn't need machine learning

You can start straight off

It is not very math heavy, mostly based on intuition. Can learn math along the way

Can choose to work on either NLP or CV

I chose CV since it interested me more



Deep Learning is the study of big neural networks and their properties

Interesting thing about neural networks which beats other algorithms - They can capture higher order relations - No need to do feature engineering



You might think neural network is just a single algorithm, but there is an ocean of research going in it

We will start with all the concepts you need to know to get started in this field



There are multiple kinds of neural networks designed for specific use-cases. Main ones being DNN for tabular data, CNN for visual or speech information, RNN LSTM and Transformers for NLP



Today our main lead for the talk is CNN



The basic concept of a CNN is to identify key features from an image and build on top of that layer-by-layer

This is done with something called filters in computer vision. 



Image is nothing but a set of pixel values i.e a 3d matrix. We can perform matrix operations on it



A filter is one which can identify a particular feature from a region of image. Like consider a 9. It consists of a straight line and a 0. 



Layer is a bunch of filters. Neural network is a bunch of filters.

![](<https://miro.medium.com/max/3288/1*uAeANQIOQPqWZnnuH-VEyw.jpeg>)

The filter works by operating on a region of image by an operation called convolution similar to element-wise production of matrices. Each filter goes across entire region of image and tries to find out a particular feature



By performing the operation we divided the information into multiple sub-parts. We can have all conv layers but it's too much in computation. So we do pooling.



Pooling is nothing but identify most important features from an image and passing it across. This is max pooling. We have a bunch of these kind of layers. 



After some time you have observed all the features, you need to make decisions out of it. That is done by using dense layers similar to a general DNN. Flatten the output and keep dense layers



Finally you constrain the outputs to a certain number of classes using something called softmax layer.



**Loss function** :-The way to tell a neural network whether it is going in right direction or wrong direction



**Backpropagation and Gradient descent**:- The magic which stitches all these together. The math is complex but you can take it for granted. 



**Learning rate** :- Speed of learning information, needs to be tuned, a hyperparameter



**Batch size** :- Number of images in an iteration



**Epoch** :- When network has looked at all the images



**Activation Functions** :- Add non-linearity to your network. Relu, sigmod, tanh.



**Optimizers** :- Fast track the convergence of neural networks. Few of them being SGD, RMSProp, Adam, SGD with Momentum



**Convolution** :- Like we have seen before. 3x3 and 1x1. There are multiple types of convolution. Depthwise convolution, Separable Convolution, Group Convolution, Transpose convolution, Dilated Convolution



# Regularization in Deep Learning



**Batch Normalization** :- Normalize input at every layer. Variants of it Layer Normalization, Weight Normalization, Instance Normalization

**Dropout** :- Make network do more with less resources

**Label smoothing** :- Real world data may be noisy. Don't be too confident

**Weight regularization** :- Simple weights means better generalization

**Data Augmentation** :- Increase your dataset with augmentation



 # Network architectures

![](<http://www.videantis.com/wp-content/uploads/2018/07/LSVRC-winners-over-time.png>)

 

This is how deep learning exploded

Network architectures are optimized neural networks built to handle image recognition tasks

Important ones :- VGG, **Resnet**, Inception, Densenet



Try using any network architecture than designing one



# Datasets for classification

MNIST

![](https://camo.githubusercontent.com/d440ac2eee1cb3ea33340a2c5f6f15a0878e9275/687474703a2f2f692e7974696d672e636f6d2f76692f3051493378675875422d512f687164656661756c742e6a7067)

CIFAR10

![](<https://miro.medium.com/max/824/1*SZnidBt7CQ4Xqcag6rd8Ew.png>)

Tinyimagenet/CIFAR100

![](<https://datarepository.wolframcloud.com/resources/images/69f/69f1e629-81e6-4eaa-998f-f6734fcd2cb3-io-4-o.en.gif>)



Once done with classification try out a task like Style Transfer

![](<https://forums.fast.ai/uploads/default/original/3X/c/f/cf5bc18c3b8a3761e341056c3f131012001d4248.jpeg>)



# Object Detection

![](<https://miro.medium.com/max/503/1*xWntyXM0W-SuDMgWMM6mCg.png>)

Multiple architectures like all variants RCNN, YOLO, SSD, Retinanet etc.

RCNN variants come under two stage detection

YOLO, SSD come under single stage detection

Datasets :- COCO, Pascal VOC

# Image segmentation

Classify each and every pixel as belonging to a class or not

FCN, Unet, Deep Lab V3, Mask RCNN

![](<https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2019/03/Screenshot-from-2019-03-28-11-45-55.png>)

Datasets :- Cityscape, Camvid, KITTI



# Transfer Learning

Transfer knowledge from one domain to another domain

Use a network trained on tons of data as feature extractor

Finetune for your usecase

# Image captioning

![](<https://miro.medium.com/max/3548/1*6BFOIdSHlk24Z3DFEakvnQ.png>)

Datasets :- MS COCO, Flickr 8k

Architectures :- CNN + LSTM, Attention models



# Generative Adversial Network

Build a basic GAN like celebrity face generation with DCGAN

Try a complex GAN like Pix2Pix or CycleGAN then

Architectures :- DCGAN, CycleGAN, Pix2Pix, CGAN, WGAN etc.

- thispersondoesnotexist.com

Datasets :- MNIST, CelebA, UTKFace

![](<https://www.lyrn.ai/wp-content/uploads/2018/12/Faces-example-1366x877.jpg>)



# Face Detection and Recognition

Architectures :- Siamese Networks Facenet

Dataset :- Aligned Face Dataset, UMDFaces

![](<https://cdn2.hubspot.net/hubfs/3873528/FACIAL-RECOGNITION-compressor.png>)



# Object Tracking

![](<https://s3-us-west-2.amazonaws.com/static.pyimagesearch.com/object-tracking-dlib/object_tracking_dlib_example01.gif>)

Architectures :- Siamese networks, GOTURN

Dataset :- MOT, Tracknet



# Mobile optimized networks

Architectures :- Mobilenet, Squeezenet, EfficientNet

Tools :- TFLite

# Action Recognition 

![](<https://pythonawesome.com/content/images/2019/10/recog_actions2.gif>)

Architectures :- CNN+LSTM, Two stream neural networks, 3D CNN

Datasets :- UCF101, Kinetics-600



# Pose Estimation

![](<https://nanonets.com/blog/content/images/2019/04/human-action-recognition.png>)

Architectures :- Deep Pose, Convolutional Pose Machines

Datasets :- LSP, FLIC



# Human Object Interaction

![](<https://i.ytimg.com/vi/j1DSo_dxANk/maxresdefault.jpg>)

Architectures :- TIN, ICAN

Datasets :- HICO



# 3D computer vision

Until now we have been looking at 2d data. But as humans we understand from 3d images. Computers do this using point clouds generated by using multiple cameras.

Architectures :- Pointnet, Shapenet

![](<http://stanford.edu/~rqi/pointnet/images/segres.jpg>)



# Projecting Work

1.  Add all the projects to github
2. Blog your work on various platform. Collaborate with publications. Post on linkedin
3. Take internships whenever possible
4. Speak about your work at events, showcase your projects

