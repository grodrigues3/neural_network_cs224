"""
Neural Network Implementation For Name Entity Recognition
Based On Programming Assignment 4
CS 224N / Ling 284
http://nlp.stanford.edu/~socherr/pa4_ner.pdf
"""
__author__ = "grodrigues3"
import math
import numpy as np
import scipy as sp
from WindowModel import *
from WordMatrix import *
import time

from sklearn.externals import joblib

dataDir = "../data/"
dataFile = dataDir +"train"
vocab = dataDir +"vocab.txt"
testFile = dataDir+"dev"
wordDim = 50

pickledParams = "../trainedParams/"

class NeuralNetwork:

    def __init__(self, contextSize=5, nOuts=1, dimHidden=100, dimIn=50):


        self.vocabMatrix = WordMatrix()
        self.contextSize = contextSize
        self.windowModel = WindowModel( testFile, self.vocabMatrix, (self.contextSize - 1)/2 )
        self.alpha = .1
        self.nOuts = nOuts
        self.dimHidden = dimHidden
        self.dimIn = dimIn


        self.e_init     = np.sqrt(6) / np.sqrt(contextSize*dimIn + dimHidden)
        self.W          = np.random.random((self.dimHidden,self. dimIn*self.contextSize)) * 2*self.e_init - self.e_init
        self.U          = np.random.random((self.dimHidden+1, 1))* 2*self.e_init - self.e_init
        self.b1         = np.random.random((self.dimHidden, 1))* 2*self.e_init - self.e_init
        self.L          = self.vocabMatrix.wordMatrix

        self.labels      =       None
        self.trainingData=       None

    def train(self, numEpochs = 10, saveToFile= False, warmStart=True):
        """
        Feed an example forward, then calculate the derivative of the loss for each of the parameters
        and update the derivative accordingly

        :param numEpochs: how many iterations of sgd
        :param saveToFile: should i write the params to a file?
        :param warmStart: start with 20 iterations already performed?
        :return:
        """
        wordTuples = self.windowModel.generate_word_tuples()
        trainingMatrix, labels = self.windowModel.generate_word_vectors()
        self.trainingData = trainingMatrix
        alpha = .001
        if warmStart:
            self.W, self.U, self.b1, self.L = joblib.load(pickledParams+'trainedPar_20.pkl')
        print "Beginning SGD...."
        s = time.time()
        for x in range(numEpochs):
            for trEx in range(trainingMatrix.shape[0]):
                ex = trainingMatrix[trEx:trEx+1,:] #stupid way to get one row without screwing up the dimensions
                label = labels[:, trEx:trEx+1]
                wT = wordTuples[trEx]
                self.labels = label
                dW, dU, db1, dL = self.dTHETA(self.W, self.U, self.b1, ex)
                self.W -= alpha * dW
                self.U -= alpha * dU
                self.b1 -= alpha * db1

                for i in range(self.contextSize):
                    wordInd = wT[i]
                    self.L[:, wordInd] -= dL[i*self.dimIn:(i+1)*self.dimIn, 0]

                if trEx%5000==0:
                    print "Epoch: {0}\t Training Example: {1:0>5d}\tElapsed Time:{2:.2f} s".format(x, trEx, time.time() - s )
        if saveToFile:
            joblib.dump([self.W, self.U, self.b1, self.L], pickledParams + 'trainedPar_new'+str(numEpochs) + '.pkl')

        #self._gradient_check()
        finalPreds = self.feedforward()
        print self.f1_score(labels, finalPreds)

    def test(self, tf = testFile, pickled=True):
        print "Testing your parameters on the test datafile (../data/dev)"
        testWindowModel = WindowModel( tf, self.vocabMatrix, (self.contextSize - 1)/2 )
        wordTuples = testWindowModel.generate_word_tuples()
        testMatrix, labels = self.windowModel.generate_word_vectors()
        self.trainingData = testMatrix
        
        #Use the 60 epoch one for now
        if pickled:
            self.W, self.U, self.b1, self.L = joblib.load(pickledParams+'trainedPar_60.pkl')
        finalPreds = self.feedforward()
        print self.f1_score(labels, finalPreds)

    def f1_score(self, trueVals, predictions):
        trueVals = trueVals.T
        predictions = predictions > .5
        truePos = 0
        falsePos = 0
        falseNeg =0

        for i,label in enumerate(trueVals):
            if label and predictions[i]:
                truePos +=1
            if not label and predictions[i]:
                falsePos +=1
            if label and not predictions[i]:
                falseNeg +=1

        recall = 1.0*truePos / (truePos + falseNeg)
        precision = 1.0*truePos / (truePos+falsePos)

        f1 = 2.0*precision*recall / (precision+recall)
        return f1


    def feedforward(self, W=None, U=None, b1=None, trEx=None):
        if W == None:
            W = self.W
        if U == None:
            U = self.U
        if b1 == None:
            b1 = self.b1
        if trEx == None:
            trEx = self.trainingData
        z = np.dot(W, trEx.T)+ b1
        a = np.tanh(z)
        bias = np.ones((1,a.shape[1]))
        myA = np.vstack( (bias, a) )

        g = self._sigmoid( np.dot(myA.T, U) )
        return g


    def _sigmoid(self, z):
        denom = 1+np.exp(-z)
        return 1.0/denom

    def costFunctionReg(self, h, y, C=.1):
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

    def dTHETA(self, W=None, U=None, b1=None, trEx = None):

        ##Forward Propagation
        z       = np.dot(W, trEx.T)+ b1
        a       = np.tanh(z)
        bias    = np.ones((1,a.shape[1]))
        a       = np.vstack( (bias, a) )
        sig     = self._sigmoid( np.dot(a.T, U) )

        ##Part 1 of the chain rule dh = dJ/dh
        dh = (self.labels.T/sig - (1-self.labels.T)/(1-sig) )*-1.

        ##dJ/dU     = tanh(W[x_bar]+b1) x [(1-y)/(1-h) - y/h ] .* sig .* (1-sig)
        ##          = a x [dh] .* sig .* (1-sig)
        ##          = (Hx1) x (1xm) x(mx1) = Hx1
        dU = np.dot(a, dh *(sig *(1-sig) ) )


        ##dJ/dW     = [U .* [1-tanh**2 ( W[x] + b1)] .* [(1-y)/(1-h) - y/h ] .* sig .* (1-sig)] x [x_bar]
        ##          = U .* (1-a**2) .* [dh] .* sig .* (1-sig) x  [x_bar]
        ##          = (Hx1) .* (Hxm) x(mxCn) = HxCn
        front   = dh*sig*(1-sig)
        middle  =  (U*(1-a*a)) * front
        dW      = np.dot(middle, trEx)[1:,:]


        ##db1       = [U .* [1-tanh**2 ( W[x] + b1)] .* [(1-y)/(1-h) - y/h ] .* sig .* (1-sig)] x [x_bar]
        ##          = U .* (1-a**2) .* [dh] .* sig .* (1-sig)
        ##          = (Hx1) .* (Hxm) .* (1 xm) = Hx1
        db1         =  middle[1:,:]


        second = U *(1-(a*a) ) #H x m
        third = np.dot(second,front) # #H x1
        dL = np.dot(W.T, third[1:,:])

        return dW, dU, db1, dL


    def dJ_dU(self,W=None, U=None, b1=None, trEx=None):
        """
        a = H x m
        sig = m x1
        dh = m x 1
        :return:
        """

        z = np.dot(W,trEx.T)+ b1
        a = np.tanh(z)
        bias = np.ones((1,a.shape[1]))
        a = np.vstack( (bias, a) )
        sig = self._sigmoid( np.dot(a.T, U) )
        dh = (self.labels.T/sig - (1-self.labels.T)/(1-sig) )*-1.
        deriv = np.dot(a, dh *(sig *(1-sig) ) )
        return deriv


    def dJ_dW(self, W=None, U=None, b1=None,trEx=None):
        z = np.dot(W, trEx.T)+ b1
        a = np.tanh(z)
        bias = np.ones((1,a.shape[1])) # 1x numTrainingExamples
        a = np.vstack( (bias, a) )
        sig = self._sigmoid( np.dot(a.T, U) )
        dh = (self.labels.T/sig - (1-self.labels.T)/(1-sig) )*-1.
        front = dh*sig*(1-sig) #m x 1
        middle =  (U*(1-a*a)) * front # H x m
        deriv = np.dot(middle, trEx)
        return deriv[1:,:]#skip the bias term

    def dJ_db(self, W=None, U=None, b1=None,trEx=None):
        z = np.dot(W, trEx.T)+ b1
        a = np.tanh(z)
        bias = np.ones((1,a.shape[1])) # 1x numTrainingExamples
        a = np.vstack( (bias, a) )
        sig = self._sigmoid( np.dot(a.T, U) )


    def dJ_dL(self, W=None, U=None, b1=None, trEx=None):
        z = np.dot(W, trEx.T)+ b1
        a = np.tanh(z)
        bias = np.ones((1,a.shape[1]))
        a = np.vstack( (bias, a) )
        sig = self._sigmoid( np.dot(a.T, U) )
        dh = (self.labels.T/sig - (1-self.labels.T)/(1-sig) )*-1.

        first = dh*sig*(1-sig) #m x 1
        #all elementwise
        second = U *(1-(a*a) ) #H x m
        third = np.dot(second,first) # #H x1
        final = np.dot(W.T, third[1:,:])
        #deriv = np.dot(np.matrix(deriv.T[0,1:]), self.W).T
        return final


    def d_theta(self, W=None, U=None, b=None, trEx =None):
        """
        dW  =   self.dJ_dW(W,U,b)
        dU  =   self.dJ_dU(W,U,b)
        db  =   self.dJ_db(W,U,b)
        dL  =   self.dJ_dL(W,U,b)
        """
        dW, dU, db, dL = self.dTHETA(W,U,b, trEx)
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
        W = np.random.random((self.dimHidden,self. dimIn*self.contextSize)) * 2*self.e_init - self.e_init
        U = np.random.random((self.dimHidden+1, 1))* 2*self.e_init - self.e_init
        b1 = np.random.random((self.dimHidden, 1))* 2*self.e_init - self.e_init
        #simulate a randomtraining example
        trEx = np.random.random( (1, self.contextSize*self.dimIn))* 2*self.e_init - self.e_init


        #W = np.random.random(self.W.shape)
        start =0
        #for i in range(W.size + U.size + b1.size + L.shape[0]*3):
        print "Parameter, Approximation, From Function, Difference, Index"
        for i in range(0, 15351):
                #calculate the gradient according to myImplementations
                dth= self.d_theta(W,U,b1,trEx)

                #calculate the approximation
                if i < W.size:
                    case = "W"
                    row,col = i/W.shape[1], i%W.shape[1]
                    W[row,col] += eps
                    plus_eps = self.feedforward(W,U,b1,trEx)
                    initial = self.costFunctionReg(plus_eps, self.labels.T,C=0)
                    W[row,col] -= 2*eps
                    minus_eps = self.feedforward(W,U,b1,trEx)
                    changed = self.costFunctionReg(minus_eps, self.labels.T,C=0)
                elif i < W.size + U.size:
                    case = "U"
                    Uind =i - W.size
                    row,col = Uind/U.shape[1], Uind%U.shape[1]
                    #print "U", col,row,i
                    U[row,col] += eps
                    plus_eps = self.feedforward(W,U,b1,trEx)
                    initial = self.costFunctionReg(plus_eps, self.labels.T,C=0)
                    U[row,col] -= 2*eps
                    minus_eps = self.feedforward(W,U,b1,trEx)
                    changed = self.costFunctionReg(minus_eps, self.labels.T,C=0)
                elif i < W.size + U.size +b1.size:
                    case = "b1"
                    bind =i - (W.size+U.size)
                    row,col = bind/b1.shape[1], bind%b1.shape[1]
                    #print "b1", col,row,i
                    b1[row,col] += eps
                    plus_eps = self.feedforward(W,U,b1,trEx)
                    initial = self.costFunctionReg(plus_eps, self.labels.T,C=0)
                    b1[row,col] -= 2*eps
                    minus_eps = self.feedforward(W,U,b1,trEx)
                    changed = self.costFunctionReg(minus_eps, self.labels.T,C=0)
                else:
                    case = "L"
                    Lind =i - (W.size+U.size + b1.size)
                    row,col = Lind/trEx.shape[1], Lind%trEx.shape[1]
                    #print "L", col,row,i
                    trEx[row,col] += eps
                    plus_eps = self.feedforward(W,U,b1,trEx)
                    initial = self.costFunctionReg(plus_eps, self.labels.T,C=0)
                    trEx[row,col] -= 2*eps
                    minus_eps = self.feedforward(W,U,b1,trEx)
                    changed = self.costFunctionReg(minus_eps, self.labels.T,C=0)

                approx = (initial - changed) / (2* eps)

                if approx-dth[0, i] > 1e-4 or i %100 ==0:
                    print case, approx, dth[0, i], approx-dth[0, i], i
                    #Just make sure the difference between my calculation and the approximation is sufficiently small
                    try:
                        assert approx-dth[0, i] < 1e-4
                    except AssertionError:
                        #for debugging
                        pdb.set_trace()


    def __str__(self):
        return "{0},{1}, {2}, {3},{4}, {5}. \n{6}".format(self.dimHidden,\
                                              self.contextSize, self.dimIn, self.alpha, \
                                              self.W.shape, self.U.shape, self.b1)


if __name__ == "__main__":
    n = NeuralNetwork()
    #n.train(numEpochs = 60)
    n.test(testFile, pickled= True)
    
