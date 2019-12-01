# TFLITE

Code available here [TFLite code](https://github.com/Anil-matcha/Speaking-Engagements/blob/master/Tf_lite.ipynb)

![](https://miro.medium.com/max/8122/1*jM9g4p6g8k3FYquSG5l1gw.jpeg)

Set of tools to optimize models for mobile

# About me

Senior computer vision engineer at Samsung R&D buliding optimized models for providing intelligence to mobile devices

![](https://www.analyticsinsight.net/wp-content/uploads/2019/07/Computer-Vision-Future-1024x682.jpg)

Reach me at Linkedin https://in.linkedin.com/in/anilmatcha

Apart from this I do blogging

Run a telegram group for computer vision professionals http://t.me/ComputerVisiongroup

2 components

1. Tensorflow Interpreter :- Capable to run optimized models on different hardware such as mobile, raspberry etc.
2. TFlite converter :- Converts the tf trained model to an optimized one to run faster



# Need for TFLite :- 

1. Speed, can use smaller models
2. Privacy
3. Internet
4. Battery

![](https://miro.medium.com/max/1548/1*MtXrCASxGrQtX2PPmhJcAw.png)

# Steps to run a model on mobile

1. Train a model on tensorflow
2. Convert the model to tflite format using converter
3. Add the tflite model to the device to run inference

![](https://miro.medium.com/max/2516/0*Bt9qwKDjd1xi5RDd.)

# Utilities :-

TFLite supports a subset of operations of tensorflow which are optimized for embedded devices. 

More ops can be added using Tensorflow Select 

You can select an operation and build a binary of TFLite which might be heavier. Also can write a custom operation in C++ if needed and can be used



# Training classifier

![](https://miro.medium.com/max/2560/1*2oSWoC8Y3s25F87kfmEIcQ.jpeg)

1. Create dataset
2. Train final layers
3. Fine-tune last layers

![](https://miro.medium.com/max/1920/1*Ww3AMxZeoiB84GVSRBr4Bw.png)

4. Fine-tune more layers

# Steps for android app

The models and the labels file needs to be added in assets folder

```
android {
    aaptOptions {
        noCompress "tflite"
    }
}
```

In build.gradle this needs to be added to say not to compress tflite files to android

![](https://miro.medium.com/max/13340/1*32DnkPuY-yP1hgDQR8Ke5g.png)

```
tflite = new Interpreter(tfliteModel, tfliteOptions);
```

This line instantiates the interpreter for android

```
imgData = ByteBuffer.allocateDirect(
            DIM_BATCH_SIZE  *
            getImageSizeX() *
            getImageSizeY() *
            DIM_PIXEL_SIZE  *
            getNumBytesPerChannel()
);
```

The image is provided like this

```
imgData.load(bitmap);
ImageProcessor imageProcessor =
    new ImageProcessor.Builder()
        .add(new ResizeWithCropOrPadOp(cropSize, cropSize))
        .add(new ResizeOp(imageSizeX, imageSizeY, ResizeMethod.BILINEAR))
        .add(new Rot90Op(numRoration))
        .add(getPreprocessNormalizeOp())
        .build();
imageProcessor.process(inputImageBuffer)        
```

To run tflite inference of an image

```
tflite.run(inputImageBuffer.getBuffer(), outputProbabilityBuffer.getBuffer());
```

# Quantization

![](https://miro.medium.com/max/1350/0*Eqk0bsuRgzVf0Fyu)

# Pruning

![](https://miro.medium.com/max/1532/0*iNI8Oc80Eunm8NgI)

https://www.tensorflow.org/model_optimization

#### Quantization aware training