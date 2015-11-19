# naiveBayes.py
# -------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# naiveBayes.py
# -------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import classificationMethod
import math
import numpy as np

class NaiveBayesClassifier(classificationMethod.ClassificationMethod):
    """
    See the project description for the specifications of the Naive Bayes classifier.

    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    """
    def __init__(self, legalLabels):
        self.legalLabels = legalLabels
        self.labelIndex = {}
        for i in range(len(legalLabels)):
            self.labelIndex[legalLabels[i]] = i
        self.type = "naivebayes"
        self.k = 1 # this is the smoothing parameter, ** use it in your train method **
        self.automaticTuning = False # Look at this flag to decide whether to choose k automatically ** use this in your train method **

    def setSmoothing(self, k):
        """
        This is used by the main method to change the smoothing parameter before training.
        Do not modify this method.
        """
        self.k = k

    def train(self, trainingData, trainingLabels, validationData, validationLabels, showPlot=None):
        """
        Outside shell to call your method. Do not modify this method.
        """

        # might be useful in your code later...
        # this is a list of all features in the training set.
        # self.features = list(set([ f for datum in trainingData for f in datum.keys() ]));
        self.features = range(trainingData[0].size)

        if (self.automaticTuning):
            kgrid = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 50]
        else:
            kgrid = [self.k]

        self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, kgrid)

    def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, kgrid):
        """
        Trains the classifier by collecting counts over the training data, and
        stores the Laplace smoothed estimates so that they can be used to classify.
        Evaluate each value of k in kgrid to choose the smoothing parameter
        that gives the best accuracy on the held-out validationData.

        trainingData and validationData are lists of feature Counters.  The corresponding
        label lists contain the correct label for each datum.

        To get the list of all possible features or labels, use self.features and
        self.legalLabels.
        """

        numDimensions = trainingData[0].size
        bestAccuracyCount = -1 # best accuracy so far on validation set

        # Common training - get all counts from training data
        # We only do it once - save computation in tuning smoothing parameter
        commonPriorArray = np.zeros(len(self.legalLabels))
        commonConditionalProbMatrix = np.zeros([len(self.legalLabels), numDimensions])


        for (datum, label) in zip(trainingData, trainingLabels):
            indx = self.labelIndex[label]
            commonPriorArray[indx] += 1
            for i in xrange(numDimensions):
                if datum[i] > 0:
                    commonConditionalProbMatrix[indx,i] +=1

        for k in kgrid: # Smoothing parameter tuning loop!
            priorArray = np.copy(commonPriorArray)
            conditionalProbMatrix = np.copy(commonConditionalProbMatrix)
            countsArray = np.copy(commonPriorArray)

            #smoothing
            conditionalProbMatrix += k
            countsArray += 2*k

            #normalizing:
            priorArray = priorArray / np.linalg.norm(priorArray,1)
            for label in self.legalLabels:
                indx = self.labelIndex[label]
                for i in xrange(numDimensions):
                    conditionalProbMatrix[indx,i] = conditionalProbMatrix[indx,i] / countsArray[indx]

            self.prior = priorArray
            self.conditionalProb = conditionalProbMatrix

            # evaluating performance on validation set
            predictions = self.classify(validationData)
            accuracyCount =  [predictions[i] == validationLabels[i] for i in range(len(validationLabels))].count(True)

            print "Performance on validation set for k=%f: (%.1f%%)" % (k, 100.0*accuracyCount/len(validationLabels))
            if accuracyCount > bestAccuracyCount:
                bestParams = (priorArray, conditionalProbMatrix, k)
                bestAccuracyCount = accuracyCount

            # end of automatic tuning loop
        self.prior, self.conditionalProb, self.k = bestParams

    def classify(self, testData):
        """
        Classify the data based on the posterior distribution over labels.

        You shouldn't modify this method.
        """
        guesses = []
        self.posteriors = [] # Log posteriors are stored for later data analysis (autograder).
        for datum in testData:
            posterior = self.calculateLogJointProbabilities(datum)
            maxIndx = np.argmax(posterior)
            guesses.append(self.legalLabels[maxIndx])
            # self.posteriors.append(posterior)
        return guesses

    def calculateLogJointProbabilities(self, datum):
        """
        Returns the log-joint distribution over legal labels and the datum.
        Each log-probability should be stored in the log-joint counter, e.g.
        logJoint[3] = <Estimate of log( P(Label = 3, datum) )>

        To get the list of all possible features or labels, use self.features and
        self.legalLabels.
        """
        numDimensions = datum.size
        logJoint = np.ones(len(self.labelIndex))

        for label in self.legalLabels:
            indx = self.labelIndex[label]
            logJoint[indx] = math.log(self.prior[indx])
            for i in xrange(numDimensions):
                if datum[i] > 0:
                    logJoint[indx] += math.log(self.conditionalProb[indx,i])
                else:
                    logJoint[indx] += math.log(1-self.conditionalProb[indx,i])

        return logJoint

#TODO: this function need to be fixed if we have more time
    # def findHighOddsFeatures(self, label1, label2):
        # """
        # Returns the 100 best features for the odds ratio:
                # P(feature=1 | label1)/P(feature=1 | label2)

        # Note: you may find 'self.features' a useful way to loop through all possible features
        # """
        # featuresOdds = []

        # for feat in self.features:
            # featuresOdds.append((self.conditionalProb[feat, label1]/self.conditionalProb[feat, label2], feat))
        # featuresOdds.sort()
        # featuresOdds = [feat for val, feat in featuresOdds[-100:]]

        # return featuresOdds
