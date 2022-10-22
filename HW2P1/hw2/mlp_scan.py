# DO NOT import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
import os
import sys

sys.path.append('mytorch')
from loss import *
from activation import *
from linear import *
from conv import *


class CNN_SimpleScanningMLP():
    def __init__(self):
        ## Your code goes here -->
        # self.conv1 = ???
        # self.conv2 = ???
        # self.conv3 = ???
        # ...
        # <---------------------
        self.conv1 = Conv1d(24, 8, 8, 4) # in_channel, out_channel, kernel_size, stride
        self.conv2 = Conv1d(8, 16, 1, 1)
        self.conv3 = Conv1d(16, 4, 1, 1)
        convs = [self.conv1, self.conv2, self.conv3]

        self.layers = []
        for i in range(3):
            self.layers.append(convs[i])
            self.layers.append(ReLU())
        self.layers = self.layers[:-1] # remove final ReLU
        self.layers.append(Flatten())

    def init_weights(self, weights):
        # Load the weights for your CNN from the MLP Weights given
        # w1, w2, w3 contain the weights for the three layers of the MLP
        # Load them appropriately into the CNN

        w1,w2,w3 = weights

        w1 = w1.T
        w2 = w2.T
        w3 = w3.T

        w1 = w1.reshape(8, 8, 24) # out_channel, kernel_size, in_channel
        w1 = np.transpose(w1, (0, 2, 1)) # out_channel, in_channel, kernel_size
        
        w2 = w2.reshape(16, 1, 8)
        w2 = np.transpose(w2, (0, 2, 1))
        
        w3 = w3.reshape(4, 1, 16)
        w3 = np.transpose(w3, (0, 2, 1))

        self.conv1.conv1d_stride1.W = w1
        self.conv2.conv1d_stride1.W = w2
        self.conv3.conv1d_stride1.W = w3

    def forward(self, A):
        """
        Do not modify this method

        Argument:
            A (np.array): (batch size, in channel, in width)
        Return:
            Z (np.array): (batch size, out channel , out width)
        """

        Z = A
        for layer in self.layers:
            Z = layer.forward(Z)
        return Z

    def backward(self, dLdZ):
        """
        Do not modify this method

        Argument:
            dLdZ (np.array): (batch size, out channel, out width)
        Return:
            dLdA (np.array): (batch size, in channel, in width)
        """

        for layer in self.layers[::-1]:
            dLdA = layer.backward(dLdA)
        return dLdA


class CNN_DistributedScanningMLP():
    def __init__(self):
        ## Your code goes here -->
        # self.conv1 = ???
        # self.conv2 = ???
        # self.conv3 = ???
        # ...
        # <---------------------
        self.conv1 = Conv1d(24, 2, 2, 2) # in_channel, out_channel, kernel_size, stride
        self.conv2 = Conv1d(2, 8, 2, 2)
        self.conv3 = Conv1d(8, 4, 2, 1)
        convs = [self.conv1, self.conv2, self.conv3]

        self.layers = []
        for i in range(3):
            self.layers.append(convs[i])
            self.layers.append(ReLU())
        self.layers = self.layers[:-1] # remove final ReLU
        self.layers.append(Flatten())

    def __call__(self, A):
        # Do not modify this method
        return self.forward(A)

    def init_weights(self, weights):
        # Load the weights for your CNN from the MLP Weights given
        # w1, w2, w3 contain the weights for the three layers of the MLP
        # Load them appropriately into the CNN

        w1,w2,w3 = weights

        w1 = w1[:48, :2]
        w2 = w2[:4, :8]

        w1 = w1.T
        w2 = w2.T
        w3 = w3.T

        w1 = w1.reshape(2, 2, 24) # out_channel, kernel_size, in_channel
        w1 = np.transpose(w1, (0, 2, 1)) # out_channel, in_channel, kernel_size
        
        w2 = w2.reshape(8, 2, 2)
        w2 = np.transpose(w2, (0, 2, 1))
        
        w3 = w3.reshape(4, 2, 8)
        w3 = np.transpose(w3, (0, 2, 1))

        self.conv1.conv1d_stride1.W = w1
        self.conv2.conv1d_stride1.W = w2
        self.conv3.conv1d_stride1.W = w3


    def forward(self, A):
        """
        Do not modify this method

        Argument:
            A (np.array): (batch size, in channel, in width)
        Return:
            Z (np.array): (batch size, out channel , out width)
        """

        Z = A
        for layer in self.layers:
            Z = layer.forward(Z)
        return Z

    def backward(self, dLdZ):
        """
        Do not modify this method

        Argument:
            dLdZ (np.array): (batch size, out channel, out width)
        Return:
            dLdA (np.array): (batch size, in channel, in width)
        """
        dLdA = dLdZ
        for layer in self.layers[::-1]:
            dLdA = layer.backward(dLdA)
        return dLdA
