Author: Garrett Rodrigues

This is a python based Neural Network Implementation for Named Entity Recognition as described in Programming Assignment 4 for CS224: 

http://nlp.stanford.edu/~socherr/pa4_ner.pdf

The model consists of a 3 layer neural network with a single hidden layer and binary output that indicates whether a given word belongs to the named entity class PERSON.  The training data consists of a word, label pairs indicating whether or not the given word belongs to the class PERSON. 


There directory structure is as follows:

    pa4-ner/

        python/: contains the code for training and testing the network
            NeuralNetwork.py: the main module containing the NeuralNetwork class, back propagation implementation and class loading
            WindowModel.py: an implementation of the sliding window used for this project
            WordMatrix.py: a class for generating the initial word matrix as well as maintaining the mapping from word to vectorized representation

        data/: contains the training and test data as well as the initializations for all of the words in the vocabulary
            train: the training data
            dev: the test data (the filename 'dev' is preserved from the Stanford course)
            vocab: a list of all of the words in the training dataset
            wordVectors: an 50-dim vector initialization for each of the words in the vocab

        trainedParams/: some parameter values from an already trained neural network so that you can train and test faster
               trainedPar_20.pkl: parameters after 20 iterations
               trainedPar_60.pkl: parameters after 60 iterations


Base Rates:
The train file contains 203621 words total
    192493 :    0.945349 
    11128  :    0.005465
