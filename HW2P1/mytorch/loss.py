import numpy as np

class MSELoss:
    
    def forward(self, A, Y):
    
        self.A = A
        self.Y = Y
        N      = A.shape[0]
        C      = A.shape[1]
        se     = np.multiply(A - Y, A - Y) # TODO
        sse = np.ones((1,N))@se@np.ones((C,1))# TODO
        mse    = sse/(N*C)
        
        return mse
    
    def backward(self):
    
        dLdA = self.A - self.Y
        
        return dLdA

class CrossEntropyLoss:
    
    def forward(self, A, Y):
    
        self.A   = A
        self.Y   = Y
        N        = A.shape[0]
        C        = A.shape[1]
        Ones_C   = np.ones((C, 1), dtype="f")
        Ones_N   = np.ones((N, 1), dtype="f")

        self.softmax     = np.exp(A)/(np.exp(A)@Ones_C@np.transpose(Ones_C)) # TODO
        crossentropy     = np.multiply(-Y, np.log(self.softmax)) # TODO
        sum_crossentropy = np.transpose(Ones_N)@crossentropy@Ones_C # TODO
        L = sum_crossentropy / N
        
        return L
    
    def backward(self):
    
        dLdA = self.softmax - self.Y # TODO
        
        return dLdA
