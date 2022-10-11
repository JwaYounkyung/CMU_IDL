import numpy as np

class Upsample1d():

    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):

        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """
        b, c, w = A.shape
        k = self.upsampling_factor
        self.w_in = w
        
        Ones = np.zeros((b, c, w*k - (k-1)), dtype=type(A[0][0][0]))

        for i in range(b):
            for j in range(c):
                for l in range(w):
                    Ones[i][j][k*l] = A[i][j][l]

        Z = Ones # TODO

        return Z

    def backward(self, dLdZ):

        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """
        
        b, c, w_out = dLdZ.shape
        k = self.upsampling_factor
        
        Ones = np.zeros((b, c, self.w_in), dtype=type(dLdZ[0][0][0]))

        for i in range(b):
            for j in range(c):
                for l in range(self.w_in):
                    Ones[i][j][l] = dLdZ[i][j][k*l]

        dLdA = Ones  #TODO

        return dLdA

class Downsample1d():

    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor

    def forward(self, A):

        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """
        b, c, w_in = A.shape
        k = self.downsampling_factor
        self.w_in = w_in
        w_out = int((w_in + k - 1)/k)
        
        Ones = np.zeros((b, c, w_out), dtype=type(A[0][0][0]))

        for i in range(b):
            for j in range(c):
                for l in range(w_out):
                    Ones[i][j][l] = A[i][j][k*l]
        Z = Ones # TODO

        return Z

    def backward(self, dLdZ):

        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """
        b, c, w_out = dLdZ.shape
        k = self.downsampling_factor
        
        Ones = np.zeros((b, c, self.w_in), dtype=type(dLdZ[0][0][0]))

        for i in range(b):
            for j in range(c):
                for l in range(w_out):
                    Ones[i][j][k*l] = dLdZ[i][j][l]

        dLdA = Ones # TODO
        return dLdA

class Upsample2d():

    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):

        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, in_channels, output_width, output_height)
        """
        b, c, w_in, h_in = A.shape
        k = self.upsampling_factor
        self.w_in = w_in
        self.h_in = h_in
        
        Ones = np.zeros((b, c, w_in*k - (k-1),  h_in*k - (k-1)), dtype=type(A[0][0][0][0]))

        for i in range(b):
            for j in range(c):
                for l in range(w_in):
                    for m in range(h_in):
                        Ones[i][j][k*l][k*m] = A[i][j][l][m]

        Z = Ones # TODO

        return Z

    def backward(self, dLdZ):

        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """

        b, c, w_out, h_out = dLdZ.shape
        k = self.upsampling_factor
        
        Ones = np.zeros((b, c, self.w_in, self.h_in), dtype=type(dLdZ[0][0][0][0]))

        for i in range(b):
            for j in range(c):
                for l in range(self.w_in):
                    for m in range(self.h_in):
                        Ones[i][j][l][m] = dLdZ[i][j][k*l][k*m]

        dLdA = Ones  #TODO

        return dLdA


class Downsample2d():

    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor

    def forward(self, A):

        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, in_channels, output_width, output_height)
        """
        b, c, w_in, h_in = A.shape
        k = self.downsampling_factor
        self.w_in = w_in
        self.h_in = h_in
        w_out = int((w_in + k - 1)/k)
        h_out = int((h_in + k - 1)/k)
        
        Ones = np.zeros((b, c, w_out, h_out), dtype=type(A[0][0][0][0]))

        for i in range(b):
            for j in range(c):
                for l in range(w_out):
                    for m in range(h_out):
                        Ones[i][j][l][m] = A[i][j][k*l][k*m]
        Z = Ones # TODO

        return Z

    def backward(self, dLdZ):

        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        
        b, c, w_out, h_out = dLdZ.shape
        k = self.downsampling_factor
        
        Ones = np.zeros((b, c, self.w_in, self.h_in), dtype=type(dLdZ[0][0][0][0]))

        for i in range(b):
            for j in range(c):
                for l in range(w_out):
                    for m in range(h_out):
                        Ones[i][j][k*l][k*m] = dLdZ[i][j][l][m]

        dLdA = Ones # TODO
        return dLdA