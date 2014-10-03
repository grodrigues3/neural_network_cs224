import scipy as sp
import numpy as np

dataDir = "../data/"
dataFile = dataDir +"train"
wordVecs = dataDir +"wordVectors.txt"
vocab = dataDir +"vocab.txt"
wordDim = 50

class WordMatrix():
    def __init__(self):
        self.indexDict = {}
        self.wordMatrix = self.generate_word_matrix()


    def generate_word_matrix(self):
        with open(wordVecs, 'r') as f, open(vocab, 'r') as g:
            for i,word in enumerate(g):
                self.indexDict[word.strip("\n")] = i
            numWords = i+1

            newMatrix = np.zeros( (wordDim,numWords))
            for i, vec in enumerate(f):
                features = np.array( vec.split())
                newMatrix[:,i] = features.T
            return newMatrix







