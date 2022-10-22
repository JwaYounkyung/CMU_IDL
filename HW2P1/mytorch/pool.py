import numpy as np
from resampling import *

class MaxPool2d_stride1():

    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        self.in_size = A.shape[-1]
        o_size = int((A.shape[-2] - self.kernel)/1) + 1
        Z = np.zeros((A.shape[0], A.shape[1], o_size, o_size))
        self.pidx = np.empty((A.shape[0], A.shape[1], o_size, o_size), dtype=object)

        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                for k in range(o_size):
                    for l in range(o_size):
                        a = A[i,j,k:k+self.kernel,l:l+self.kernel]
                        self.pidx[i,j,k,l] = np.unravel_index(a.argmax(), a.shape)
                        Z[i,j,k,l] = np.max(a)
        
        return Z
    
    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        dLdA = np.zeros((dLdZ.shape[0], dLdZ.shape[1], self.in_size, self.in_size))

        for i in range(dLdZ.shape[0]):
            for j in range(dLdZ.shape[1]):
                for k in range(dLdZ.shape[2]):
                    for l in range(dLdZ.shape[3]):
                        dLdA[i,j,k+self.pidx[i,j,k,l][0],l+self.pidx[i,j,k,l][1]] += dLdZ[i,j,k,l]

        return dLdA

class MeanPool2d_stride1():

    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        self.in_size = A.shape[-1]
        o_size = int((A.shape[-2] - self.kernel)/1) + 1
        Z = np.zeros((A.shape[0], A.shape[1], o_size, o_size))

        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                for k in range(o_size):
                    for l in range(o_size):
                        a = A[i,j,k:k+self.kernel,l:l+self.kernel]
                        Z[i,j,k,l] = np.mean(a)
        
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        dLdA = np.zeros((dLdZ.shape[0], dLdZ.shape[1], self.in_size, self.in_size))

        for i in range(dLdZ.shape[0]):
            for j in range(dLdZ.shape[1]):
                for k in range(dLdZ.shape[2]):
                    for l in range(dLdZ.shape[3]):
                        dLdA[i,j,k:k+self.kernel,l:l+self.kernel] += dLdZ[i,j,k,l]/(self.kernel**2)

        return dLdA

class MaxPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride
        
        #Create an instance of MaxPool2d_stride1
        self.maxpool2d_stride1 = MaxPool2d_stride1(kernel) #TODO
        self.downsample2d = Downsample2d(stride) #TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        stride1 = self.maxpool2d_stride1.forward(A) 
        Z = self.downsample2d.forward(stride1)
        
        return Z
    
    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        dLdZ = self.downsample2d.backward(dLdZ)
        dLdA = self.maxpool2d_stride1.backward(dLdZ)
        
        return dLdA

class MeanPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        #Create an instance of MaxPool2d_stride1
        self.meanpool2d_stride1 = MeanPool2d_stride1(kernel) #TODO
        self.downsample2d = Downsample2d(stride) #TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        stride1 = self.meanpool2d_stride1.forward(A)
        Z = self.downsample2d.forward(stride1)
        
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        dLdZ = self.downsample2d.backward(dLdZ)
        dLdA = self.meanpool2d_stride1.backward(dLdZ)

        return dLdA