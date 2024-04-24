"""
Homework 5 - CNNs
CS1430 - Computer Vision
Brown University
"""

import tensorflow as tf
from keras.layers import \
       Conv2D, MaxPool2D, Dropout, Flatten, Dense
# from keras.optimizers import experimental

import hyperparameters as hp


class YourModel(tf.keras.Model):
    """ Your own neural network model. """

    def __init__(self):
        super(YourModel, self).__init__()

        # TASK 1
        # TODO: Select an optimizer for your network (see the documentation
        #       for tf.keras.optimizers)
       
        self.optimizer = tf.keras.optimizers.experimental.SGD(
       learning_rate=hp.learning_rate,
       momentum=hp.momentum,
       name='SGD')

        self.architecture = [

             # 2 convolutional layers with activation functions
              Conv2D(64, 3, 1, padding="same",
                   activation="relu", name="block1_conv1"),
            Conv2D(64, 3, 1, padding="same",
                   activation="relu", name="block1_conv2"),
            MaxPool2D(2, name="block1_pool"),
            # Block 2
            Conv2D(128, 3, 1, padding="same",
                   activation="relu", name="block2_conv1"),
            Conv2D(128, 3, 1, padding="same",
                   activation="relu", name="block2_conv2"),
            MaxPool2D(2, name="block2_pool"),
            # Block 3
            Conv2D(256, 3, 1, padding="same",
                   activation="relu", name="block3_conv1"),
            Conv2D(256, 3, 1, padding="same",
                   activation="relu", name="block3_conv2"),
            Conv2D(256, 3, 1, padding="same",
                   activation="relu", name="block3_conv3"),
            MaxPool2D(2, name="block3_pool"),
            # Block 4
            Conv2D(256, 3, 1, padding="same",
                   activation="relu", name="block4_conv1"),
            Conv2D(256, 3, 1, padding="same",
                   activation="relu", name="block4_conv2"),
            Conv2D(256, 3, 1, padding="same",
                   activation="relu", name="block4_conv3"),
            MaxPool2D(2, name="block4_pool"),
            # Flatten to vectorize activations
            Flatten(name="flatten"),
            # dense layers
            Dense(100, name="block1_dense1", activation="relu"),
            # Adding regularization via dropout layer
            Dropout(0.1),
            Dense(50, name="block1_dense2", activation="relu"),
            # A final dense layer with the number of classes as the number of neurons (or logits) + softmax activation to produce a probability distribution
            Dense(units=hp.num_classes, activation='softmax', name='block1_dense3'),
        ]

    def call(self, x):
        """ Passes input image through the network. """

        for layer in self.architecture:
            x = layer(x)

        return x

    @staticmethod
    def loss_fn(labels, predictions):
        """ Loss function for the model. """
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)(labels, predictions)
        return loss


class VGGModel(tf.keras.Model):
    def __init__(self):
        super(VGGModel, self).__init__()

        self.optimizer = tf.keras.optimizers.experimental.SGD(
       learning_rate=hp.learning_rate,
       momentum=hp.momentum,
       name='SGD')


        # Don't change the below:

        self.vgg16 = [
            # Block 1
            Conv2D(64, 3, 1, padding="same",
                   activation="relu", name="block1_conv1"),
            Conv2D(64, 3, 1, padding="same",
                   activation="relu", name="block1_conv2"),
            MaxPool2D(2, name="block1_pool"),
            # Block 2
            Conv2D(128, 3, 1, padding="same",
                   activation="relu", name="block2_conv1"),
            Conv2D(128, 3, 1, padding="same",
                   activation="relu", name="block2_conv2"),
            MaxPool2D(2, name="block2_pool"),
            # Block 3
            Conv2D(256, 3, 1, padding="same",
                   activation="relu", name="block3_conv1"),
            Conv2D(256, 3, 1, padding="same",
                   activation="relu", name="block3_conv2"),
            Conv2D(256, 3, 1, padding="same",
                   activation="relu", name="block3_conv3"),
            MaxPool2D(2, name="block3_pool"),
            # Block 4
            Conv2D(512, 3, 1, padding="same",
                   activation="relu", name="block4_conv1"),
            Conv2D(512, 3, 1, padding="same",
                   activation="relu", name="block4_conv2"),
            Conv2D(512, 3, 1, padding="same",
                   activation="relu", name="block4_conv3"),
            MaxPool2D(2, name="block4_pool"),
            # Block 5
            Conv2D(512, 3, 1, padding="same",
                   activation="relu", name="block5_conv1"),
            Conv2D(512, 3, 1, padding="same",
                   activation="relu", name="block5_conv2"),
            Conv2D(512, 3, 1, padding="same",
                   activation="relu", name="block5_conv3"),
            MaxPool2D(2, name="block5_pool")
        ]
        for layer in self.vgg16.layers:
              layer.trainable = False

        self.head = [
            Flatten(name="flatten"),
            Dense(500, activation="relu", name="fc1"),
            Dense(200, activation="relu", name="fc2"),
            Dense(15, activation="softmax", name="predictions")
        ]

        # Don't change the below:
        self.vgg16 = tf.keras.Sequential(self.vgg16, name="vgg_base")
        self.head = tf.keras.Sequential(self.head, name="vgg_head")

    def call(self, x):
        """ Passes the image through the network. """

        x = self.vgg16(x)
        x = self.head(x)

        return x

    @staticmethod
    def loss_fn(labels, predictions):
        """ Loss function for model. """

        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)(labels, predictions)
        return loss
