# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

from audioop import bias
import numpy as np
from resampling import *

class Conv1d_stride1():
    def __init__(self, in_channels, out_channels, kernel_size,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(0, 1.0, (out_channels, in_channels, kernel_size))
        else:
            self.W = weight_init_fn(out_channels, in_channels, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        self.A = A
        self.in_size = A.shape[-1]
        convs = []
        o_size = int((A.shape[-1] - self.kernel_size)/1) + 1 # stride 1

        for i in range(o_size):
            a = A[:,:,i:self.kernel_size+i]
            conv = np.tensordot(a, self.W, [(1,2), (1,2)])
            conv = conv + self.b
            convs.append(conv)

        convs = np.array(convs)
        convs = np.transpose(convs, (1, 2, 0))

        Z = convs # TODO
        
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        # dLdW conv
        convs = []
        for i in range(self.W.shape[-1]):
            a = self.A[:,:,i:dLdZ.shape[-1]+i]
            conv = np.tensordot(dLdZ, a, [(0,2), (0,2)])
            convs.append(conv)
        
        convs = np.array(convs)
        convs = np.transpose(convs, (1, 2, 0))
        
        self.dLdW = convs # TODO
        self.dLdb = np.sum(dLdZ, axis=(0,2)) # TODO

        # dLdA conv
        pad_dLdZ = np.pad(dLdZ, ((0,0),(0,0),(self.kernel_size-1, self.kernel_size-1)), 'constant')
        flip_W = np.flip(self.W, 2)

        convs = []
        for i in range(self.in_size):
            dldz = pad_dLdZ[:,:,i:self.kernel_size+i]
            conv = np.tensordot(dldz, flip_W, [(1,2), (0,2)])
            convs.append(conv)
        
        convs = np.array(convs)
        convs = np.transpose(convs, (1, 2, 0))
        dLdA = convs # TODO

        return dLdA

class Conv1d():
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names
    
        self.stride = stride

        # Initialize Conv1d() and Downsample1d() isntance
        self.conv1d_stride1 = Conv1d_stride1(in_channels, out_channels, kernel_size, weight_init_fn, bias_init_fn) # TODO
        self.downsample1d =  Downsample1d(stride)# TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """

        # Call Conv1d_stride1
        # TODO
        stride1 = self.conv1d_stride1.forward(A)

        # downsample
        Z = self.downsample1d.forward(stride1) # TODO

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        # Call downsample1d backward
        # TODO
        dLdZ = self.downsample1d.backward(dLdZ)

        # Call Conv1d_stride1 backward
        dLdA = self.conv1d_stride1.backward(dLdZ) # TODO

        return dLdA


class Conv2d_stride1():
    def __init__(self, in_channels, out_channels,
                 kernel_size, weight_init_fn=None, bias_init_fn=None):

        # Do not modify this method

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(0, 1.0, (out_channels, in_channels, kernel_size, kernel_size))
        else:
            self.W = weight_init_fn(out_channels, in_channels, kernel_size, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        self.A = A
        self.in_size = A.shape[-1]
        convs = []
        o_size = int((A.shape[-1] - self.kernel_size)/1) + 1 # stride 1

        for i in range(o_size):
            convs_j = []
            for j in range(o_size):
                a = A[:,:,i:self.kernel_size+i, j:self.kernel_size+j]
                conv = np.tensordot(a, self.W, [(1,2,3), (1,2,3)])
                conv = conv + self.b
                convs_j.append(conv)
            convs.append(convs_j)

        convs = np.array(convs)
        convs = np.transpose(convs, (2, 3, 0, 1))
        Z = convs #TODO

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        # dLdW conv
        convs = []
        for i in range(self.W.shape[-2]):
            convs_j = []
            for j in range(self.W.shape[-1]):
                a = self.A[:,:,i:dLdZ.shape[-2]+i, j:dLdZ.shape[-1]+j]
                conv = np.tensordot(dLdZ, a, [(0,2,3), (0,2,3)])
                convs_j.append(conv)
            convs.append(convs_j)
        
        convs = np.array(convs)
        convs = np.transpose(convs, (2, 3, 0, 1))

        self.dLdW = convs # TODO        
        self.dLdb = np.sum(dLdZ, axis=(0,2,3)) # TODO

        # dLdA conv
        pad_dLdZ = np.pad(dLdZ, ((0,0),(0,0),(self.kernel_size-1, self.kernel_size-1),(self.kernel_size-1, self.kernel_size-1)), 'constant')
        flip_W = np.flip(self.W, (2,3))

        convs = []
        for i in range(self.in_size):
            convs_j = []
            for j in range(self.in_size):
                dldz = pad_dLdZ[:,:,i:self.kernel_size+i, j:self.kernel_size+j]
                conv = np.tensordot(dldz, flip_W, [(1,2,3), (0,2,3)])
                convs_j.append(conv)
            convs.append(convs_j)
        
        convs = np.array(convs)
        convs = np.transpose(convs, (2, 3, 0, 1))
        dLdA = convs # TODO

        return dLdA

class Conv2d():
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names
        self.stride = stride

        # Initialize Conv2d() and Downsample2d() isntance
        self.conv2d_stride1 = Conv2d_stride1(in_channels, out_channels, kernel_size, weight_init_fn, bias_init_fn) # TODO
        self.downsample2d = Downsample2d(stride) # TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        # Call Conv2d_stride1
        # TODO
        stride1 = self.conv2d_stride1.forward(A)

        # downsample
        Z = self.downsample2d.forward(stride1) # TODO

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """

        # Call downsample1d backward
        # TODO
        dLdZ = self.downsample2d.backward(dLdZ)

        # Call Conv1d_stride1 backward
        dLdA = self.conv2d_stride1.backward(dLdZ) # TODO

        return dLdA

class ConvTranspose1d():
    def __init__(self, in_channels, out_channels, kernel_size, upsampling_factor,
                weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.upsampling_factor = upsampling_factor

        # Initialize Conv1d stride 1 and upsample1d isntance
        #TODO
        self.upsample1d = Upsample1d(upsampling_factor) #TODO
        self.conv1d_stride1 = Conv1d_stride1(in_channels, out_channels, kernel_size, weight_init_fn, bias_init_fn) #TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        #TODO
        # upsample
        A_upsampled = self.upsample1d.forward(A) #TODO

        # Call Conv1d_stride1()
        Z = self.conv1d_stride1.forward(A_upsampled) #TODO

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        #TODO

        #Call backward in the correct order
        delta_out = self.conv1d_stride1.backward(dLdZ) #TODO
        dLdA = self.upsample1d.backward(delta_out) #TODO

        return dLdA

class ConvTranspose2d():
    def __init__(self, in_channels, out_channels, kernel_size, upsampling_factor,
                weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.upsampling_factor = upsampling_factor

        # Initialize Conv2d() isntance
        self.conv2d_stride1 = Conv2d_stride1(in_channels, out_channels, kernel_size, weight_init_fn, bias_init_fn) #TODO
        self.upsample2d = Upsample2d(upsampling_factor) #TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        # upsample
        A_upsampled = self.upsample2d.forward(A) #TODO

        # Call Conv2d_stride1()
        Z = self.conv2d_stride1.forward(A_upsampled) #TODO

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        #Call backward in correct order
        delta_out = self.conv2d_stride1.backward(dLdZ) #TODO
        dLdA = self.upsample2d.backward(delta_out) #TODO

        return dLdA

class Flatten():

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, in_width)
        Return:
            Z (np.array): (batch_size, in_channels * in width)
        """
        self.in_channels = A.shape[1]
        self.in_width = A.shape[2]
        Z = A.reshape(A.shape[0], -1) # TODO

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch size, in channels * in width)
        Return:
            dLdA (np.array): (batch size, in channels, in width)
        """
        dLdA = dLdZ.reshape(dLdZ.shape[0], self.in_channels, self.in_width) # TODO

        return dLdA