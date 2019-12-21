# Deep dive into Cyclegan

Computer vision engineer at Samsung R&D buliding optimized models for providing intelligence to mobile devices

**Usecases** :- Gallery object search, human object interaction, live segmentation, scene recognition

![img](https://camo.githubusercontent.com/9389472d239c0d04dca439ab87294942f4165f3e/68747470733a2f2f7777772e616e616c7974696373696e73696768742e6e65742f77702d636f6e74656e742f75706c6f6164732f323031392f30372f436f6d70757465722d566973696f6e2d4675747572652d31303234783638322e6a7067)

Reach me out at Linkedin https://in.linkedin.com/in/anilmatcha

I run a telegram group for computer vision professionals http://t.me/ComputerVisiongroup

# What is GAN

# Generative Algorithms

- We all are familiar with task such as classification
- Neural networks are pretty good at doing calculations, making comparisons
- Now they are able to imagine things which is considered only capable by humans
- Humans are pretty good at imagining just by closing eyes
- A generative algorithm trained on learning how an object say horse looks like can generate new horse images which look like a horse in real

GAN is one popular generative algorithm

![](<https://blog.paperspace.com/content/images/2019/10/dzone.png>)

# Applications

Few applications

Generating art for cartoons or video games

Used for augmenting input data

Image-to-image translation

![](<https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2019/06/Example-of-Four-Image-to-Image-Translations-Performed-with-CycleGAN.png>)

Text to Image conversion

![](<https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2019/06/Example-of-Textual-Descriptions-and-GAN-Generated-Photographs-of-Birds.png>)

Super resolution

![](<https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2019/06/Example-of-GAN-Generated-Images-with-Super-Resolution.png>)

Faceapp like image editing

![](<https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2019/04/Capture.png>)

Music generation

Deepfakes

More at https://github.com/nashory/gans-awesome-applications

# GAN working

- Consists of 2 networks instead of one network
- Generator task is to generate images
- Discriminator looks at an image and says if it's fake or real
- Both the networks learn simultaneously by playing a game of overpowering each other
- Training stops when discriminator gets confused completely
- Many popular architectures like DCGAN, StyleGAN, CGAN, BigGAN etc.

Let's discuss **CycleGAN**

- CycleGAN is used in domain transfer usecases like zebra-horse, day-night, photo-painting

![](<https://blog.paperspace.com/content/images/2019/10/cyclegan.jpg>)

- Basically an Image-to-Image translation problem
- Pix2Pix does this by having paired images from domain 1 to domain 2
- Cyclegan works without paired images with a clever trick

![](<https://shreyas-kowshik.github.io/blog/figs/2019-8-18/cyclegan_arch.jpg>)

- Instead of a single generator-discriminator we now have a pair of generator discriminator
- First generator translates the image to a new domain
- Second generator brings the image to the previous domain  
- The task of both the discriminators is to validate the generated images



# Loss Function

**Discriminator Loss**

- General discriminator works like a classification network
- Classifies the input as real or fake class basically a 2 class network
- Cross-entropy loss can be used directly
- CycleGAN uses a PatchGAN discriminator taken from pix2pix
- Instead of a single output it produces a grid NxN of outputs
- Each output in the grid corresponds to a patch of certain size in the input image
- Now instead of classifying entire image we are classifying patches of images
- Produced better and sharper features while doing research in pix2pix
- Take loss for each cell and sum up or use mse

**Generator Loss**

- Adversial loss, Cyclic Loss, Identity loss

- **Adverisal Loss** :- 

  Opposite of Discriminator loss

- **Cyclic Loss** :- 

  Image translated to a new domain and translated back should be similar.

  L1 loss is applied. 

  What this does is it not only translates the input image to a new domain 

  But also keeps few relevant features of original domain

  So a horse converted to zebra looks like original horse

- **Identity Loss** :-

  Image of opposite domain should not be altered

  Mainly used for Photo generation from paintings to preserve color composition between the input and output

​       ![](<https://www.researchgate.net/profile/Jun_Yan_Zhu/publication/322060135/figure/fig3/AS:631630267949070@1527603799396/The-effect-of-the-identity-mapping-loss-on-Monets-painting-photos-From-left-to-right.png>)



# Face Changing

**UTKFace Dataset**

![](<https://susanqq.github.io/UTKFace/icon/samples.png>)

age_gender_race_date&time.jpg

Collect all 20-30 age images into folderA

Collect all 50-60 age images into folderB

Train CycleGAN and do domain transfer 

Discriminator produces a patch of input_size/4

Renset-style generator 

**Code Walkthrough**

https://tinyurl.com/cyclegan

# Tips to stabilize training

1.  Use progressive resizing introduced by Jeremy Howard.  Training with 256x256 would constrain you to smaller batch size like 1. Start with say 64x64(allows batch size 32) and progressively increase using checkpoints. Same like training a classification network
2. Look at loss values closely. See which one is lacking by looking at the images and change based on that. Reconstruction is bad -> Boost cyclic loss. Generated images are bad -> Boost adversial loss. Color lost in generated images -> Boost identity loss



# Debugging

In a classification network you check progress with say accuracy 

In a gan accuracy goes up and down, so look at the generated images

Generate images at every few iterations and see results

If you feel something is missing, tweak the losses and repeat



# Extension

Make the generator do multiple tasks by using conditions

Can be done in multiple ways

Add an extra channel to RGB input like a channel of all 0's for age conversion and all 1's for race conversion

Have multiple discriminators one to discriminate age and one to discriminate race

Train together both the tasks

Instead can have a single discriminator by passing the conditional input to discriminator as well just like discriminator

Instead of passing the condition as a channel, can pass the condition through an embedding layer and concatenating to the dense layer



# Applying on real-world images

Took a face recognition model to get the face cropped it out and sent to generator and replaced it back

![](<https://raw.githubusercontent.com/Anil-matcha/Face-aging-and-race-change-with-conditional-cycle-gan/master/data/full_pics/big3.jpg>)

![](<https://raw.githubusercontent.com/Anil-matcha/Face-aging-and-race-change-with-conditional-cycle-gan/master/data/full_pics/big3_black.jpg>)



More resources for different kinds of GAN :- https://github.com/hindupuravinash/the-gan-zoo

# Q & A

1. Changing parameters so as to affect a certain type of object in the image. ?

   We saw that using conditional gan

2. How many iterations to run ?

   No hard figure

3. Using GANs in our Image recognition platform ?

   Some startups are using and seems to have decent results

4. Production usage in gan ?

   Faceapp and similar applications use GAN in production

5. Mode collapse and GAN metrics such as FID score ?

   To deal with mode collapse use WGAN

   Inception score and FID score are used to evaluate GAN performance

   Inception score uses Inception network.

   ![](<https://miro.medium.com/max/1860/1*X29oOi1Tzch2j6MuG9XS1Q.png>)

   ![](<https://miro.medium.com/max/2380/1*23gj_d3dxfm5FoKae_pc5Q.png>)

   FID improves on IS by comparing generated statistics with real data statistics of an Inception network

6. GAN transfer leanring ?

   Not yet

7. Learning path ?

   Start with simple DCGAN MNIST dataset, then celebA dataset