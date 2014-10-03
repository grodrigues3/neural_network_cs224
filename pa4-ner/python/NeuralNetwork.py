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
        self.U = np.random.random((self.dimHidden+1, 1))
        self.b1 = np.random.random((self.dimHidden, 1))
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


        self.trainingData = train[:batchSize,:]
        self.labels = labels[:,:batchSize]
        print self.labels
        print self.costFunctionReg(, self.labels, 0)
        #print "hypothesis", n.feedforward().shape
        #print "dh", n.dJ_dh(W,U,b1).shape
        #print "U", self.dJ_dU(W,U,b1).shape
        #print "W", self.dJ_dW(W,U,b1).shape
        #print "b1", self.dJ_db(W,U,b1).shape
        #print "L", self.dJ_dL(W,U,b1).shape

        # print "grad_check",
        # X = self._gradient_check()
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
        a = np.vstack( (bias, a) )
        #keep this for later usage
        self.a = a
        g = self._sigmoid( np.dot(a.T, U) )
        self.g = g
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
        return self.dh

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
        dh = self.labels.T/sig - (1-self.labels.T)/(1-sig)
        deriv = np.dot(a, dh *(sig *(1-sig) ) )
        return deriv


    def dJ_dW(self, W=None, U=None, b1=None):
        z = np.dot(W, self.trainingData.T)+ b1
        a = np.tanh(z)
        bias = np.ones((1,a.shape[1]))
        a = np.vstack( (bias, a) )
        sig = self._sigmoid( np.dot(a.T, U) )
        dh = self.labels.T/sig - (1-self.labels.T)/(1-sig)
        scale = float(np.dot( dh.T, (sig* (1-sig)) ))
        deriv = scale*self.U * (1-(a*a))
        deriv = np.dot(deriv, self.trainingData)
        return deriv[1:,:]#skip the bias term

    def dJ_db(self, W=None, U=None, b1=None):
        z = np.dot(W, self.trainingData.T)+ b1
        a = np.tanh(z)
        bias = np.ones((1,a.shape[1]))
        a = np.vstack( (bias, a) )
        sig = self._sigmoid( np.dot(a.T, U) )
        dh = self.labels.T/sig - (1-self.labels.T)/(1-sig)

        scale = float(np.dot( dh.T, (sig* (1-sig)) ))
        deriv = scale*np.dot(self.U.T, (1-np.dot(a, a.T))).T
        return deriv #skip the bias term

    def dJ_dL(self, W=None, U=None, b1=None):
        z = np.dot(W, self.trainingData.T)+ b1
        a = np.tanh(z)
        bias = np.ones((1,a.shape[1]))
        a = np.vstack( (bias, a) )
        sig = self._sigmoid( np.dot(a.T, U) )
        dh = self.labels.T/sig - (1-self.labels.T)/(1-sig)

        scale = float(np.dot( dh.T, (sig* (1-sig)) ))
        #all elementwise
        deriv = scale*self.U * (1-(a*a))
        deriv = np.dot(np.matrix(deriv.T[0,1:]), self.W).T
        return deriv


    def d_theta(self, W=None, U=None, b=None):
        print W.sum()
        print U.sum()
        print b.sum()

        dW  =   self.dJ_dW(W,U,b)
        dU  =   self.dJ_dU(W,U,b)
        db  =   self.dJ_db(W,U,b)
        dL  =   self.dJ_dL(W,U,b)

        dW  =   dW.reshape( (1, dW.size))
        dU  =   dU.reshape( (1, dU.size))
        db  =   db.reshape( (1, db.size))
        dL  =   dL.reshape( (1, dL.size))

        dtheta = np.hstack( (dW, dU, db, dL))

        return dtheta




    def _gradient_check(self):
        numPerts= self.W.size #+ self.b1.size + self.U.size
        eps = .00001
        W = np.zeros(self.W.shape)
        U = np.zeros(self.U.shape)
        b1 = np.zeros(self.b1.shape)

        return self.d_theta(W,U,b1)
        """
        for i in range(W):
            for j in range(W):
                predictions = self.feedforward(W,U,b1)
                initial = self.costFunctionReg(predictions, self.labels,C=0)
                diffW = self.W.copy()
                diffW[i,j] += eps
                predictions1 = self.feedforward(diffW)
                changed = self.costFunctionReg(predictions1, self.labels,C=0)
                approx = (changed - initial) / eps
        """





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