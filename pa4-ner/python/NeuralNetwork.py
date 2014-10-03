import math
import numpy as np
import scipy as sp
from WindowModel import *
from WordMatrix import *


dataDir = "../data/"
dataFile = dataDir +"train"
vocab = dataDir +"vocab.txt"
wordDim = 50

class NeuralNetwork:

    def __init__(self, nIns=3, nOuts=1, dimHidden=100, dimIn=50):
        self.vocabMatrix = WordMatrix()
        self.alpha = .1
        self.nIns = nIns
        self.nOuts = nOuts
        self.dimHidden = dimHidden
        self.dimIn = dimIn

        self.e_init = np.sqrt(6) / np.sqrt(nIns*dimIn + dimHidden)
        self.W = np.random.random((self.dimHidden,self. dimIn*self.nIns)) * 2*self.e_init - self.e_init
        self.U = np.random.random((self.dimHidden+1, 1))* 2*self.e_init - self.e_init
        self.b1 = np.random.random((self.dimHidden, 1))* 2*self.e_init - self.e_init
        self.L = self.vocabMatrix.wordMatrix

        self.trainingData   =       None
        self.labels         =       None


    def train(self, train, labels):
        """
        :param train: m x nC
        :param labels: m x 1 (or m x numOutputNodes)
        :return:
        """
        batchSize = 10

        W   = self.W
        U   = self.U
        b1   = self.b1


        self.trainingData = train[:1,:]
        #self.trainingDat = np.zeros(self.trainingData.shape)
        self.labels = labels[:,:1]
        #print self.labels
        #print self.costFunctionReg(, self.labels, 0)
        #print "hypothesis", n.feedforward().shape
        #print "dh", n.dJ_dh(W,U,b1).shape
        #print "U", self.dJ_dU(W,U,b1).shape
        #print "W", self.dJ_dW(W,U,b1).shape
        #print "b1", self.dJ_db(W,U,b1).shape
        #print "L", self.dJ_dL(W,U,b1).shape

        print "grad_check",
        X = self._gradient_check()
        # print X.shape
        # for x in range(W.size):
        #     print X[0,x]
        # for y in range(U.size):
        #     print X[0,x+y]
        #
        # for z in range(b1.size):
        #     print X[0,x+y+z]
        #
        # for zz in range(L.size):
        #     print X[0,x+y+z+zz]


    def feedforward(self, W=None, U=None, b1=None):
        if W == None:
            W = self.W
        if U == None:
            U = self.U
        if b1 == None:
            b1 = self.b1
        z = np.dot(W, self.trainingData.T)+ b1
        a = np.tanh(z)
        bias = np.ones((1,a.shape[1]))
        myA = np.vstack( (bias, a) )

        g = self._sigmoid( np.dot(myA.T, U) )
        return g


    def _sigmoid(self, z):
        denom = 1+np.exp(-z)
        return 1.0/denom

    def costFunctionReg(self, h, y, C=.1):
        """
        h : m x 1
        y : m x 1
        hT * y = scalar
        """
        m = h.shape[0]
        first = (np.log(h)).T.dot(-y)
        second = (np.log(1-h)).T.dot(1-y)
        J = 1.0/m* (first - second)
        # Add in regularization
        reg = C/(2.0*m) * np.dot(self.U[1:].T, self.U[1:])
        reg += C/(2.0*m) * np.sum(np.dot(self.W.T, self.W))
        J += reg
        return J[0][0]



    ###########################
    #DERIVATIVE CALCULATIONS
    ###########################

    def dJ_dh(self,W,U,b1):
        g = self.feedforward(W,U,b1)
        self.dh = self.labels.T/g - (1-self.labels.T)/(1-g)
        return -1*self.dh

    def dJ_dU(self,W=None, U=None, b1=None):
        """
        a = H x m
        sig = m x1
        dh = m x 1
        :return:
        """

        z = np.dot(W, self.trainingData.T)+ b1
        a = np.tanh(z)
        bias = np.ones((1,a.shape[1]))
        a = np.vstack( (bias, a) )
        sig = self._sigmoid( np.dot(a.T, U) )
        dh = (self.labels.T/sig - (1-self.labels.T)/(1-sig) )*-1.
        deriv = np.dot(a, dh *(sig *(1-sig) ) )
        return deriv


    def dJ_dW(self, W=None, U=None, b1=None):
        z = np.dot(W, self.trainingData.T)+ b1
        a = np.tanh(z)
        bias = np.ones((1,a.shape[1])) # 1x numTrainingExamples
        a = np.vstack( (bias, a) )
        sig = self._sigmoid( np.dot(a.T, U) )
        dh = (self.labels.T/sig - (1-self.labels.T)/(1-sig) )*-1.
        front = dh*sig*(1-sig) #m x 1
        middle =  (U*(1-a*a)) * front # H x m
        deriv = np.dot(middle, self.trainingData)
        return deriv[1:,:]#skip the bias term

    def dJ_db(self, W=None, U=None, b1=None):
        z = np.dot(W, self.trainingData.T)+ b1
        a = np.tanh(z)
        bias = np.ones((1,a.shape[1])) # 1x numTrainingExamples
        a = np.vstack( (bias, a) )
        sig = self._sigmoid( np.dot(a.T, U) )
        dh = (self.labels.T/sig - (1-self.labels.T)/(1-sig) )*-1.
        front = dh*sig*(1-sig) #m x 1
        middle =  (U* ((1-a*a) * front)) # H x m
        return middle[1:,:] #skip the bias term

    def dJ_dL(self, W=None, U=None, b1=None):
        z = np.dot(W, self.trainingData.T)+ b1
        a = np.tanh(z)
        bias = np.ones((1,a.shape[1]))
        a = np.vstack( (bias, a) )
        sig = self._sigmoid( np.dot(a.T, U) )
        dh = (self.labels.T/sig - (1-self.labels.T)/(1-sig) )*-1.

        first = dh*sig*(1-sig) #m x 1
        #all elementwise
        try:
            second = U *(1-(a*a) ) #H x m
            third = np.dot(second,first) # #H x1
            final = np.dot(W.T, third[1:,:])
        except:
            pdb.set_trace()
        #deriv = np.dot(np.matrix(deriv.T[0,1:]), self.W).T
        return final


    def d_theta(self, W=None, U=None, b=None):

        dW  =   self.dJ_dW(W,U,b)
        dU  =   self.dJ_dU(W,U,b)
        db  =   self.dJ_db(W,U,b)
        dL  =   self.dJ_dL(W,U,b)

        dW  =   dW.reshape( (1, dW.size))
        dU  =   dU.reshape( (1, dU.size))
        db  =   db.reshape( (1, db.size))
        dL  =   dL.reshape( (1, dL.size))

        dtheta = np.hstack( (dW, dU, db, dL))

        return np.asarray(dtheta)




    def _gradient_check(self):
        """
        """

        print "Checking the gradient calculations"
        eps = .0001

        #Pick some random values for the paramters (except for L)
        W = np.random.random((self.dimHidden,self. dimIn*self.nIns)) * 2*self.e_init - self.e_init
        U = np.random.random((self.dimHidden+1, 1))* 2*self.e_init - self.e_init
        b1 = np.random.random((self.dimHidden, 1))* 2*self.e_init - self.e_init



        #W = np.random.random(self.W.shape)
        start =0
        #for i in range(W.size + U.size + b1.size + L.shape[0]*3):
        for i in range(15351-150, 15351):
                #calculate the gradient according to myImplementations
                dth= self.d_theta(W,U,b1)

                #calculate the approximation
                if i < W.size:
                    case = "W"
                    row,col = i/W.shape[1], i%W.shape[1]
                    W[row,col] += eps
                    plus_eps = self.feedforward(W,U,b1)
                    initial = self.costFunctionReg(plus_eps, self.labels.T,C=0)
                    W[row,col] -= 2*eps
                    minus_eps = self.feedforward(W,U,b1)
                    changed = self.costFunctionReg(minus_eps, self.labels.T,C=0)
                elif i < W.size + U.size:
                    case = "U"
                    Uind =i - W.size
                    row,col = Uind/U.shape[1], Uind%U.shape[1]
                    #print "U", col,row,i
                    U[row,col] += eps
                    plus_eps = self.feedforward(W,U,b1)
                    initial = self.costFunctionReg(plus_eps, self.labels.T,C=0)
                    U[row,col] -= 2*eps
                    minus_eps = self.feedforward(W,U,b1)
                    changed = self.costFunctionReg(minus_eps, self.labels.T,C=0)
                elif i < W.size + U.size +b1.size:
                    case = "b1"
                    bind =i - (W.size+U.size)
                    row,col = bind/b1.shape[1], bind%b1.shape[1]
                    #print "b1", col,row,i
                    b1[row,col] += eps
                    plus_eps = self.feedforward(W,U,b1)
                    initial = self.costFunctionReg(plus_eps, self.labels.T,C=0)
                    b1[row,col] -= 2*eps
                    minus_eps = self.feedforward(W,U,b1)
                    changed = self.costFunctionReg(minus_eps, self.labels.T,C=0)
                else:
                    case = "L"
                    Lind =i - (W.size+U.size + b1.size)
                    row,col = Lind/L.shape[1], Lind%L.shape[1]
                    #print "L", col,row,i
                    self.trainingData[row,col] += eps
                    plus_eps = self.feedforward(W,U,b1)
                    initial = self.costFunctionReg(plus_eps, self.labels.T,C=0)
                    self.trainingData[row,col] -= 2*eps
                    minus_eps = self.feedforward(W,U,b1)
                    changed = self.costFunctionReg(minus_eps, self.labels.T,C=0)

                approx = (initial - changed) / (2* eps)
                print "Parameter, Approximation, From Function, Difference, Index"
                if approx-dth[0, i] > 1e-4 or i %100 ==0:
                    print case, approx, dth[0, i], approx-dth[0, i], i







    def __str__(self):
        return "{0},{1}, {2}, {3},{4}, {5}. \n{6}".format(self.dimHidden,\
                                              self.nIns, self.dimIn, self.alpha, \
                                              self.W.shape, self.U.shape, self.b1)


if __name__ == "__main__":
    wordMat = WordMatrix()
    wm = WindowModel(dataFile, wordMat,1)

    ##Lookup happens here
    trainingMatrix, labels = wm.generate_word_vectors()

    n = NeuralNetwork()
    n.train(trainingMatrix, labels)
    """
    x = np.random.random((150,1))
    y = np.random.randint(low=0, high =2, size=(150,1))
    one = x[:50]
    two = x[50:100]
    three = x[100:]
    n.train((one,two,three), np.ones((1,1)))
    """
    #print n.costFunctionReg(x, y)