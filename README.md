Author: Garrett Rodrigues

This is a python based Neural Network Implementation for Named Entity Recognition as desribed in Programming Assignment 4 for CS224: 

http://nlp.stanford.edu/~socherr/pa4_ner.pdf


The model consists of a 3 layer neural network with a single hidden layer and binary output that indicates whether a given word belongs to the named entity class PERSON.  The training data consists of a word, label pairs indicating whether or not the given word belongs to the class PERSON. 


There directory structure is as follows:

    
    pa4-ner/
        python/: contains the code for training and testing the network
        data/: contains the training and test data as well as the initializations for all of the words in the vocabulary
        trainedParams/: some parameter values from an already trained neural network so that you can test faster

