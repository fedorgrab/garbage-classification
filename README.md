# Garbage-Recognition
Capstone project for NEU EAI6000 - Identify the type of garbage using Neural Networks

__Group members:__
1. Fedor Grab
2. Salman Rafiullah
3. Jing Lou
4. Chandra Maddikunta

__Goal:__
Classify which type of garbage is provided on the image. This is an image recognition task.

__Dataset:__
[Publicy available](https://www.kaggle.com/asdasdasasdas/garbage-classification)

It consists of 6 different classes of garbage:
1. Cardboard (393)
1. Glass (491)
1. Metal (400)
1. Paper (584)
1. Plastic (472)
1. Trash (127)

__Summary:__
There are a total of 2467 images.

To solve this task it is a good idea to use [Transfer Learning (TL)](https://machinelearningmastery.com/how-to-use-transfer-learning-when-developing-convolutional-neural-network-models/). TL is a technique when already developed and trained models for a one type of classification problem are used to solve different classification problem.

Three of the most popular models are as follows:

1. VGG (e.g. [VGG16](https://www.kaggle.com/keras/vgg16) or [VGG19](https://www.kaggle.com/keras/vgg19)).
1. GoogLeNet (e.g. [InceptionV3](https://software.intel.com/en-us/articles/inception-v3-deep-convolutional-architecture-for-classifying-acute-myeloidlymphoblastic)).
1. Residual Network (e.g. [ResNet50](https://www.kaggle.com/keras/resnet50)).

__Plan:__
1. Load dataset of garbage images
1. Prepare data:
    1. Change the input shapes of data to be consistent
    1. Normilize it
    1. Prepare classification labels array
    1. Split data on train and split sets
    1. Implement a few different models and try to fit model to it
1. Choose the best model
1. Classify test data and count model accuracy
1. Summarize the work with some insights, explain how it is could be implemented in a real-world problem

__Objective:__
By the end of the project we expect a Neural Network model classifying a garbage to one of the 6 given above garbage type with accuracy more than 90%. It can be a complicated challenge due to the fact that provided dataset seems to be difficult to classify.
