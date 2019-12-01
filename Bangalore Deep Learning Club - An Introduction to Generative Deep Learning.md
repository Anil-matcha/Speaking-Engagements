### An Introduction to Generative Deep Learning

#### A getting started guide to Generative Deep Learning Algorithms

**Generative** and **Discriminative** models are two different approaches that are widely studied in task of classification. They follow a different route from each other to achieve the final result. **Discriminative** models are widely popular and are used more comparatively to perform the task since they give better results when provided with a good amount of data. All the popular algorithms such as **SVM, KNN** etc. and popular network architectures such as **Resnet, Inception** etc. come under this.

The task of a **discriminative** model is simple, if it is shown data from different classes it should be able to discriminate between them. For example if I show the model a set of **dog and cat images** it should be able to say what is a dog and what is a cat by using discriminative features such as **eyes shape, ears size** etc.

![img](https://cdn-images-1.medium.com/max/800/0*Lqqh5KtEA3ZKaGy5.gif)

**Generative** model on the other hand has a much more complex task to perform. It has to understand the distribution from which the data is obtained and then needs to use this understanding to perform the task of classification. Also generative models have the **capability of creating data** similar to the training data it received since it has learnt the distribution from which the data is provided. For example if I show a generative model a set of dog and cat images now the model should understand completely what are the features that belongs to a certain class and how they can be used to generate similar images. Using this information it can do multiple things. It can compare the attributes to classify the image similar to how discriminative algorithm can classify an image. It can **generate** a new image which looks like one of the class images it has been provided for training.

> **Humans don’t act like pure discriminators, we possess enormous generative capabilities**

Progress in generative algorithms is important because humans don’t act just like pure discriminators, we have enormous generative or imaginative capabilities. If we give certain attributes such as **blue car on road** we can instantly generate a picture of that in our mind and we are looking at providing this kind of intelligence to machines.

In the below content we will discuss about two famous generative algorithms **Variational Autoencoders** and **Generative Adversial Networks**

### Autoencoder

![img](https://cdn-images-1.medium.com/max/800/0*ZSa-g5wH5a6KGrIs.jpeg)

An **autoencoder** is a type of ANN used to learn efficient data codings in an unsupervised manner. The aim of an autoencoder is to learn a representation (encoding) for a set of data, typically for **dimensionality reduction.**

We as humans are pretty good at **visualizing** using few attributes. For example if we describe a human as tall, fair, bulky, no mustache, punjabi you can create a visualization based on these attributes. An autoencoder tries to achieve same thing. If I show an image of a person it learns all the attributes(known as **latent attributes**) such as the above needed to identify the person and then can use them to visualize/reconstruct the person.

![img](https://cdn-images-1.medium.com/max/800/0*ARH-HFfnLSJQV51q.png)

Autoencoder consists of 3 components

1. **Encoder**
2. **Bottleneck**
3. **Decoder**

![img](https://cdn-images-1.medium.com/max/800/0*fmfbfhHHJbkmhP0l.png)

**Encoder** is similar to any classification neural network such as **Resnet** etc. sans the prediction softmax layer. If we see the below figure of VGG network if we remove the final softmax layer the final 1000 values that we get can be thought of as 1000 latent attributes of an image.

![img](https://cdn-images-1.medium.com/max/800/0*rYUj6xEmlQDfVOXq.png)

**Decoder** is the opposite of encoder, it takes the latent attributes from the output of encoder and tries to reconstruct the image. This is done using deconv layers which can unsample the input

![img](https://cdn-images-1.medium.com/max/800/0*DyrAHiGm_bCBxOe-.png)

**Bottleneck** is the latent vector that is output by the encoder and is upsampled by the decoder. It contains the **latent attributes** that are produced by the decoder such as the height, weight etc. described above.

The network of encoder and decoder is trained together using **backpropagation** to reduce the loss of reconstruction such as mean square error between the pixels.

**Applications of autoencoder :-**

1. **Denoising images** :- Autoencoders can be used to remove noise from images. Since autoencoder learns the latent representation and not the noise it can remove the noise and give the clear image. It is trained by providing noisy images at input and we try to minimize reconstruction error with proper images at output
2. **Recommender systems** :- Netflix movie recommendation challenge winner used deep autoencoders
3. **Compression** :- As we have seen autoencoder converts an input to its latent space attributes and converts it back, it can be used for compression by making latent space much smaller in comparison to input.
4. **Dimensionality Reduction** :- Autoencoders can be used similar to **PCA** to reduce the feature space by mapping input to latent attributes and using them for modelling.
5. **Generation of data** :- A variant of autoencoders called **variational autoencoders** can be used to generate data similar to the distribution it is trained on which we will discuss below.

### Variational Autoencoder

Above we have seen how can we use an autoencoder to compress the input to its latent variables. One problem which we haven’t yet solved with the above approach is the network could learn a representation which works but doesn’t generalize well. This is a classic problem in deep neural networks and is called **overfitting**. If a neural network has enough capacity it can just memorize the input data and map it to latent attributes without creating a general understanding in which case the **latent attributes don’t possess good properties** like the one we have discusses above and they could be meaningless , the task is still performed i.e reconstruction but the latent attributes don’t carry any meaning. You can compare this to a student who has **mugged up all the answers** in the textbook and can solve the problem if given directly from textbook but completely falters even if there is a slight change in the problem. We try to solve this problem using **Variational Autoencoders** which generalize much better in comparison to Vanilla Autoencoder.

![img](https://cdn-images-1.medium.com/max/800/0*s9dNyBEtiUBBaXjL.png)

Above is an image of MNIST data trained using Vanilla Autoencoder. As we can see there are different distinct clusters formed which is what we asked the autoencoder to do. Now if we see into the figure we can clearly see that the clusters are **not continuous** and there are gaps in between. So if we take a point from the gap and pass it to the decoder it might give an output which doesn’t resemble any of the classes. We don’t want this to happen, we want the space to be continuous and the outputs to make sense. We achieve this by using **VAE**.

We want **VAE** to have the below 2 properties

1. **Continuity** :- Two close points in latent space space should give identical outputs, if not the case it means there is high variance and hence overfitting and no generalization
2. **Completeness** :- A point from latent space should map to a meaningful output and shouldn’t give an unknown image as output

![img](https://cdn-images-1.medium.com/max/800/0*tHKyxWh3om6Nokk8.png)

To achieve this VAE **encoder** part outputs along with a set of latent attributes a set of **mean and variance** corresponding to each attribute in latent space. Vanilla autoencoder encodes the input to a single set of latent attributes but **VAE** encodes each latent attribute to a distribution having a mean and variance. Each time an image is passed, a set of latent attributes are **sampled** according to their mean and variance and is passed on to the decoder. The decoder works similar to that of decoder present Vanilla autoencoder i.e it **upsamples** the latent attributes to recover the input image.

The advantage of following the above approach is since an input is mapped to a **distribution** of latent attributes, **points which are close in the latent space get mapped to similar output** by default. We enforce few **constraints** on the distribution of each latent space attribute such that it regularizes as per our expectations

i) Distribution follows a normal distribution with **variance** of each attribute **close to 1**. This prevents the clusters from becoming very **tight** and hence helps in making the latent space continuous, if not the VAE can push the cluster to a very tight group, think of like a single point, which would fail our expectation of **continuity**.

ii) We try to keep the **mean** all the clusters to be **close to 0** such that we can ensure a smooth transition from one cluster to another and there are no gaps in between since this will bring all the clusters closer to each other. This way any point in latent space maps to a meaningful output.

![img](https://cdn-images-1.medium.com/max/800/0*U16PT6BL2WlzXdy4.png)

These are the steps followed by a VAE

> Encoder receives the input and outputs a set of means and variances corresponding to each latent attribute

> A latent attribute is sampled randomly from each mean and variance and is passed to decoder

> Decoder takes the randomly sampled latent attributes, upsamples it to try and reconstruct the output by minimizing the loss of reconstruction

### Loss Function

Loss function of VAE consists of 2 parts

1) **Reconstruction Loss** :- Similar to that of Vanilla AE we use MSE or cross entropy

2) **Regularization Loss** :- We try to model the output probability distributions of each latent attribute close to a standard normal. We do this by reducing **KL Divergence** between the output probability distribution and standard normal distribution.

**Loss = Reconstruction_loss + c \* KL_loss**

**KL Divergence** is used to measure the divergence between two probability distributions. Lower the value better is the match between two distributions. **c** is a **hyperparameter** which needs to be tuned and is used to balance the importance of reconstruction loss and regularization loss.

Now we come to the interesting part of learning about using VAE to **generate new data** similar to the training data and **Applications of VAE**. Once we have trained a VAE to a good extent we should have developed a continuous and complete latent space. Now we can pick any point from the latent space and pass it to the decoder, it will **generate a new image** completely unseen till now but still looks like it belongs to the distribution of data the VAE is trained on i.e it looks like one of the classes of training data which is awesome since now the network can generate data on its own.

**Applications of VAE** :-

1. Generating new data similar to the distribution of data the VAE is trained on
2. Adding artifacts to an existing image. For example if we know the images without sunglasses on and if we know the images with sunglasses on we can take a difference between their means and use it to add a sunglasses to any new image

![img](https://cdn-images-1.medium.com/max/800/0*m33eTklemM0VePJ3.png)

### Generative Adversial Networks

GAN’s are another set of generative algorithms and are one of the primary reasons for producing so much hype in deep learning. Several applications have been made using GANs and a multitude of architectures have been researched upon which led to rapid development in the field of GAN’s which can generate cool results which can make one wonder if it is real image or an image generated by GAN. For example below are the faces of person who have never existed in the real world. Looks pretty **cool** right. You can go [**https://thispersondoesnotexist.com**](https://thispersondoesnotexist.com) . This website gives a realistic fake person on every refresh.

![img](https://cdn-images-1.medium.com/max/800/0*OtfXkDOfog4uNE3Z.jpg)

Recently Samsung published a paper in which a neural network takes just a picture and can produce a small video gif out of it. Through this they have got **Monalisa** alive. Think about what is in the future possibility. You can make you dead ancestors speak to you. Holy **awesome**

![img](https://cdn-images-1.medium.com/max/800/0*wReOpDWFSyLwbnCp.gif)

Until now we used to believe we can have faith of whatever we see or listen since they happened in real i.e **videos news must be true. Not anymore**. Now realistic fake videos or audio of the person can be generated like below. Now you can’t even trust video news.

![img](https://cdn-images-1.medium.com/max/800/0*dB95f4pXZT6pKNAL.gif)

You can do domain transfer using a GAN as well. If you have an image of a horse you can **reimagine** as how it would look like if it is a zebra by using a GAN.

![img](https://cdn-images-1.medium.com/max/800/0*vuVen2RN5ioMVNME.gif)

You can reimagine yourself playing out a protagonist character in a movie just like the below **guy transformed him into Leonardo Decaprio**. This is the next level of **Dubsmash**.

![img](https://cdn-images-1.medium.com/max/800/0*hGCOMPgw_mJg994h.gif)

The possibilities are endless. You can create an entire movie without any real cast. Take a scene from real world and convert it to anime. Create fake persons for acting as models for donning the dresses in ecommerce website.

#### Working of GAN

Now we will look into the theory behind how all this magic works. GAN consists of 2 neural networks(VAE from above has only a single neural network) which work with each other namely **Generator** and **Discriminator** . They act like **teacher-student, thug-cop**. The task of a Generator is simple as it name says it generates data for example an image which has to look like the real-world data. The task of Discriminator is to look at the data from Generator and discriminate it from real-world data i.e it should look at data generated from Generator and say it’s **fake**.

![img](https://cdn-images-1.medium.com/max/800/0*0tfrMJqOpUSoapOA.jpg)

Now the **cat and mouse game** starts. Since discriminator says that the data is fake generator tries to better itself so that it can produce more realistic data which the discriminator can’t judge as real or fake. Once this happens the discriminator knows that it is failing to properly discriminate so it will try to improve itself and next time it judges better. Now the ball is in the court of generator and this game of trying to overpower each other continues until a stage comes where the discriminator is completely confused whether the data from generator is real or fake. Now generator wins and we have been rooting for generator all along. But remember the **hero is as good as the villain**. **Avengers Endgame** was such a success because **Thanos** was such a menacing villain. So we want the discriminator to be the best and the generator needs to beat discriminator at its peak since then the victory is more sweeter.

There are multiple architectures and quite a lot of complex loss functions to make the GANs work and we will be looking at one of the most successful architecure **DCGAN** (Deep Convolutional GAN) which first introduced the usage of convolutional layers for GAN.

As said before GAN consists of two neural networks **Discriminator** and **Generator** . The architectures of these neural networks are similar to that of **Encoder** and **Decoder** in **VAE**.Discriminator is the neural network we are fairly familiar with, image as input which is sampled down with convolutional layers and finally we apply a softmax to get the output class, in this case we have only 2 classes **Fake or Not-Fake** . So common architectures like Resnet, Inception can used to model a Discriminator of DCGAN. Discriminator is trained by providing the real images as real class category and fake images given by generator as fake class category. So it is a 2 class classification problem.

![img](https://cdn-images-1.medium.com/max/800/0*OCGg7QKjYapcn8va.png)

**Generator** architecture looks opposite to that of Discriminator. It takes a linear vector and upsamples it similar to **Decoder in VAE**. The linear vector is generator by random sampling and just like in VAE we can think of it as attributes of latent space. Different random samplings generate different outputs.

![img](https://cdn-images-1.medium.com/max/800/0*YsQmo0POSVoIjzJZ.png)

Both the discriminator and generator are trained simultaneously in a **minimax** game. Discriminator tries to reduce the discriminative loss such as cross-entropy loss and Generator tries to oppose it by trying to increase the error. Unlike the general tasks like classification, detection etc. the **loss doesn’t constantly decrease** since there are two opposing parties involved. Also we don’t want either discriminator to overpower generator from the start or vice-versa since in that case it will be a one-sided game and the two networks on a whole wouldn’t be learning as there is no competition, so we start from a stage where both are equally dumb i.e think of generator as generating random images and discriminator as randomly classifying images as fake or real. The networks slowly improve by competing with each other until it reaches a stage where generator completely fools the discriminator.

Training GANs is an art and there are a lot of hacks involved in stabilizing the process of training two networks simultaneously but that is the content for another article.

**About me :- I work as a Senior Research Engineer in Samsung R&D and my area of work is computer vision, mobile object detection and recognition. You can reach out to me at** [**https://www.linkedin.com/in/anilmatcha/**](https://www.linkedin.com/in/anilmatcha/)