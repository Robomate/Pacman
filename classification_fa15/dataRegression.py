# dataRegression.py
# -----------------
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


# dataRegression.py
# -----------------
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


# This file contains feature extraction methods and harness
# code for data classification

import sys
import numpy as np
import plotUtil
import pacmanPlot
import graphicsUtils
import linearLearning

TEST_SET_SIZE = 20


# def analysis(classifier, guesses, testLabels, testData, rawTestData, printImage):
    # """
    # This function is called after learning.
    # Include any code that you want here to help you analyze your results.

    # Use the printImage(<list of pixels>) function to visualize features.

    # An example of use has been given to you.

    # - classifier is the trained classifier
    # - guesses is the list of labels predicted by your classifier on the test set
    # - testLabels is the list of true labels
    # - testData is the list of training datapoints (as util.Counter of features)
    # - rawTestData is the list of training datapoints (as samples.Datum)
    # - printImage is a method to visualize the features
    # (see its use in the odds ratio part in runClassifier method)

    # This code won't be evaluated. It is for your own optional use
    # (and you can modify the signature if you want).
    # """

    # # Put any code here...
    # # Example of use:
    # # for i in range(len(guesses)):
    # #     prediction = guesses[i]
    # #     truth = testLabels[i]
    # #     if (prediction != truth):
    # #         print "==================================="
    # #         print "Mistake on example %d" % i
    # #         print "Predicted %d; truth is %d" % (prediction, truth)
    # #         print "Image: "
    # #         print rawTestData[i]
    # #         break


## =====================
## You don't have to modify any code below.
## =====================


def default(str):
    return str + ' [Default: %default]'

USAGE_STRING = """
  USAGE:      python dataRegression.py <options>
  EXAMPLES:   (1) python dataRegression.py
                  - trains the least-square regressor on the simple data set,
                  which is randomly sampled from the linear Gaussian Model:
                  y = ax + b + G(m, var)
                  using the default 100 training examples and then test the
                  regressor on test data
                  The default training method is the analytical solution.
              (2) python dataRegression.py -d BerkleyHousing -p -t 200 -m gradient -i 30
                  -would run the least-square regressor on the Berkeley housing
                  data set, with the stochastic gradient decent method. It will
                  also print the regression result and errors during the training.
                 """


def readCommand( argv ):
    "Processes the command used to run from the command line."
    from optparse import OptionParser
    parser = OptionParser(USAGE_STRING)

    parser.add_option('-d', '--data', help=default('Dataset to use'), choices=['simple6','simple15', 'simple1000', 'BerkeleyHousing'], default='simple6')
    # parser.add_option('-t', '--training', help=default('The size of the training set'), default=100, type="int")
    # parser.add_option('-v', '--validation', help=default('The size of the validation set'), default=100, type="int")
    parser.add_option('-m', '--method', help=default('Analytical method or gradient method'), choices=['analytical', 'gradient'], default='analytical')
    parser.add_option('-p', '--Print', help=default('Print the data and weights or not'), default=False, action="store_true")
    parser.add_option('-g', '--ghosts', help=default('Plot ghosts as points'), default=True, action="store_true")
    parser.add_option('-i', '--iterations', help=default("Maximum iterations to run training"), default=30, type="int")
    parser.add_option('-s', '--test', help=default("Amount of test data to use"), default=TEST_SET_SIZE, type="int")

    options, otherjunk = parser.parse_args(argv)
    if len(otherjunk) != 0: raise Exception('Command line input not understood: ' + str(otherjunk))
    args = {}

    # Set up variables according to the command line input.
    print "Doing Regression"
    print "--------------------"
    print "data:\t\t" + options.data
    print "Regressor:\t\t" + "LinearRegression"
    if(options.data!="simple6" and options.data!="simple15" and options.data!="simple1000" and options.data!="BerkeleyHousing"):
        print "Unknown dataset", options.data
        print USAGE_STRING
        sys.exit(2)

    data = np.load(options.data + ".npz")
    print "training set size:\t" + str(len(data["data"]))

    if options.method!="analytical" and options.method!="gradient":
        print "Illegal training method"
        print USAGE_STRING
        sys.exit(2)

    regressor = linearLearning.LinearRegression()
    args['regressor'] = regressor

    return args, options

# Main harness code

def runRegressor(args, options):
    regressor = args['regressor']

    # Load data
    numIter = options.iterations

    if(options.data!="BerkeleyHousing"):
        print "Loading simple dataset: " + options.data + " ..."
        data = np.load(options.data + ".npz")
        regressor.setLearningRate(0.01)
        trainingData = data["data"]
        trainingRegressionResult = data["regressionResults"]
        validationData = data["dataVali"]
        validationRegressionResult = data["regressionResultsVali"]
        testingData = data["dataTest"]
        testingRegressionResult = data["regressionResultsTest"]
        paras = data["paras"]
    else:
        print "Loading Berkeley housing dataset ..."
        regressor.setLearningRate(0.00000001)
        data = np.load("BerkeleyHousing.npz")
        # dataTest = np.load("berkeleyHousingTest.npz")
        trainingData = data["data"]
        trainingRegressionResult = data["regressionResults"]
        validationData = []
        validationRegressionResult = []
        testingData = data["dataTest"]
        testingRegressionResult = data["regressionResults"]

    # Append 1 to all data points to allow for a bias offset (and convert to Nx2 matrix)
    trainingData = np.hstack((trainingData[:,None], np.ones( (trainingData.size,1) )))
    if options.data!="BerkeleyHousing":
        validationData = np.hstack((validationData[:,None], np.ones( (validationData.size,1) )))
    testingData = np.hstack((testingData[:,None], np.ones( (testingData.size,1) )))

    # Conduct training and testing
    print "Training..."
    if(options.method == "analytical"):
        regressor.trainAnalytical(trainingData, trainingRegressionResult)
    else:
        regressor.trainGradient(trainingData, trainingRegressionResult, numIter, options.Print)
    if options.Print:
        if options.ghosts:
            pacmanDisplay = pacmanPlot.PacmanPlotRegression();
            pacmanDisplay.plot(trainingData, trainingRegressionResult, regressor.weights, title='Training: Linear Regression')
            graphicsUtils.sleep(3)
        else:
            plotUtil.plotRegression(trainingData,trainingRegressionResult, regressor.weights,1,True,False,'Training: Linear Regression')

    if len(validationData) > 0:
        print "Validating..."
        if options.Print:
            if options.ghosts:
                pacmanDisplay = pacmanPlot.PacmanPlotRegression();
                pacmanDisplay.plot(validationData, validationRegressionResult, regressor.weights, title='Validating: Linear Regression')
                graphicsUtils.sleep(3)
            else:
                plotUtil.plotRegression(validationData,validationRegressionResult, regressor.weights,1,True,False, 'Validating: Linear Regression')
        validationLoss = regressor.regressionLoss(validationData, validationRegressionResult)
        print "Validation loss: " + str(validationLoss)
    else:
        print "No validation data provided"

    print "Testing..."
    if options.Print:
        if options.ghosts:
            pacmanDisplay = pacmanPlot.PacmanPlotRegression();
            pacmanDisplay.plot(testingData, testingRegressionResult, regressor.weights, title='Testing: Linear Regression')
            graphicsUtils.sleep(3)
        else:
            plotUtil.plotRegression(testingData, testingRegressionResult, regressor.weights,1,True,False, 'Testing: Linear Regression')

    testingLoss = regressor.regressionLoss(testingData, testingRegressionResult)
    print "Testing loss: " + str(testingLoss)

    if options.Print:
        if options.ghosts:
#             pacmanDisplay.takeControl()
            graphicsUtils.end_graphics()

if __name__ == '__main__':
    # Read input
    args, options = readCommand( sys.argv[1:] )
    # Run classifier
    runRegressor(args, options)
