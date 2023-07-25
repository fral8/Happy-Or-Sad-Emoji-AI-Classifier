# Emotion AI Classifier (Happy or Sad Emoji)

## Overview

This repository contains code for an Emotion AI Classifier that can detect whether an image is associated with a happy or sad emoji. The model is built using TensorFlow and Keras and employs a Convolutional Neural Network (CNN) architecture for image classification. The dataset used for training consists of images labeled as either happy or sad.

## Requirements

To run the code in this repository, ensure you have the following dependencies installed:

- Python 3.x
- TensorFlow
- Keras
- Matplotlib

## Dataset

The dataset used for training the emotion AI classifier is located in the 'data' directory. It includes a collection of images, each associated with either a happy or sad emoji label. The images have been preprocessed and resized to a resolution of 150x150 pixels.

## Code Structure

1. `image_generator(path)`: This function creates an image data generator using the TensorFlow ImageDataGenerator class. It loads images from the specified directory and prepares them for training. The images are rescaled to a range of [0, 1], and the generator yields batches of 10 images at a time.

2. `create_conv_model()`: This function builds the CNN model for the emotion AI classifier. The model consists of several Convolutional and Dense layers with ReLU and Sigmoid activations, respectively. It uses binary cross-entropy as the loss function and stochastic gradient descent (SGD) as the optimizer.

3. `MyCallback`: This is a custom callback class that monitors the training progress. It stops the training process early if the accuracy reaches 99.5% or higher.

4. `print_loss(history)`: This function plots the training loss over the epochs, providing a visualization of the model's training performance.

## Training

To train the emotion AI classifier, run the code in the main section. The dataset will be loaded, and the model will be trained for 500 epochs. The training process will stop early if the accuracy exceeds 99.5%. After training, the loss and accuracy curves will be displayed.

Feel free to modify the code and experiment with different hyperparameters or extend the model for other emotion classifications.

For any questions or suggestions, please contact [Francesco Alotto](mailto:franalotto94@gmail.com). Happy coding! 😊
