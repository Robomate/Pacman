# linearLearning.py
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


#
# Code for linear regression and linear classification
#

# import util
import math
# import collections
import numpy as np
import plotUtil
import pacmanPlot
import graphicsUtils
import random
import util
PRINT = False

def quadLoss(x, y):
    """
    Question 1
    
    Quadtratic loss function. Takes a scalar input x and a scalar label y
    and returns the square of the difference between them.

    Note: Do NOT add a factor of 1/2 in front of this loss (as often seen 
    with quadratic loss functions)    
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

def der_quadLoss_dx(x, y):
    """
    Question 2
    
    Derivative of the quadtratic loss function (quadLoss) with respect to the scalar input x,
    given the scalar label y.

    Note: The quad loss function does NOT add a factor of 1/2 in front of that loss (as often seen 
    with quadratic loss functions)
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

def der_dot_dw(x, weights):
    """
    Question 2
    
    The derivative of the dot product of arrays x and weights with respect to the
    weights. Returns an array of the derivative with respect to each weights term
    [der_dot_dw1, der_dot_dw2, ...]
    
    Hint: You may not need all of the input arguments.
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

def stochasticGradientDescentUpdate(datum, label, weights, alpha, der_loss_dw):
    """
    Question 2
    
    Implement the weight update equation for general stochastic gradient descent,
    given a single data point, datum, and its label and a function for the derivative 
    of the loss function with respect to the weights.
    Returns the updated weights.
    
    datum: input data point
    label: true label for data point
    weights: current weight array
    alpha: learning rate (gradient descent step size)
    
    der_loss_dw: Function for the derivative of the loss function with respect to the
    weights. Function signature: der_loss_dw(datam, label, weights), returns an array 
    of the derivative of the loss function with respect to each self.weights term
    [der_loss_dw1, der_loss_dw2, ...]
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()
    
    return weights

def sigmoid(x):
    """
    Question 3
    
    Return the output of the sigmoid function with scalar input x.
    
    x: float input to function.

    Note that this is just the sigmoid function and there are no dot
    products with weights involved.
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

def der_sigmoid_dx(x):
    """
    Question 3
    
    Derivative of the sigmoid function with respect to input x.
    Return the derivative evalutated at the input value x.
    
    x: float input to function
    
    Hint: Find (look-up) a form of the derivative that can take
    advantage of the sigmoid function you already implemented.  
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

def softmax(x):
    """
    Question 4
    
    Softmax function that takes and array of N inputs and returns an array
    N outputs.
    
    x: numpy ndarray with N entries
    Returns: ndArray with N entries
    
    Hint: Please take advantage of the numpy.exp function that takes an numpy 
    array with N inputs and applies the natural exponential function to each 
    of them, returning an numpy array with the corresponding N output values.
    
    Note that this is just the sigmoid function and there are no dot
    products with weights involved.
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

def der_softmax_dx(x, i, j):
    """
    The (i,j) entry of the Jacobian of the softmax function, dy_i/dx_j
    """
    if i == j:
        y = sigmoid(x[i])
        return y*(1-y)
    else:
        return -sigmoid(x[i])*sigmoid(x[j])

def crossEntLoss(px, py):
    # Check if input is an array
    if isinstance(px, np.ndarray ):
        loss = 0
        for (x,y) in zip(px,py):
            loss += -y*math.log(x)
    else:
        # For the scalar case, assume just two classes
        loss = -py*math.log(px)-(1-py)*math.log(1-px)
    return loss

def der_crossEntLoss_dx(px, py):
    # Check if input is an array
    if isinstance(px, np.ndarray ):
        dloss = -py*1.0/px
    else:
        # For the scalar case, assume just two classes
        dloss = -py*1.0/px-(1-py)*1.0/(1-px)

    return dloss

def crossEntLossLabel(px, label):
    """
    The cross entropy loss assuming P(Y=label) = 1 or
    if px is a scalar, we assume the case of binary variables
    and label is zero or one
    """
    # Check if input is an array
    if isinstance(px, np.ndarray ):
        loss = -math.log(px[label])
    else:
        # For the scalar case, assume just two classes and px is P(y=1|x)
        if label == 1:
            loss = -math.log(px)
        else:
            loss = -math.log(1-px)

    return loss

def der_crossEntLossLabel_dxlabel(px, label):
    # Check if input is an array
    if isinstance(px, np.ndarray ):
        dloss = -1.0/px[label]
    else:
        # For the scalar case, assume just two classes
        dloss = -label*1.0/px - (1-label)*1.0/(1-px)

    return dloss

class LinearRegression:
    """
    Basic Least-squares Regression
    """
    def __init__(self):
        print 'Linear regression initializing ...'
        self.alpha = 0.001
        self.weights = np.zeros(2)

    def setLearningRate(self, alpha):

        self.alpha = alpha

    def trainAnalytical(self, trainingData_x, trainingData_y):
        """
        Question 1
        
        Return the analytical solutions for the weights that minimize
        the quadratic loss between the input trainingData_x and 
        trainingData_y labels. 
        
        trainingData_x is a two dimensional (Nx2) array, where each row is a training point [x, 1].
        trainingData_y is a one dimensional (Nx1) array containing the scalar output
        label y for each training point
        """

        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

        self.weights = weights

    def trainGradient(self, trainingData, regressionData, numIterations, showPlot=True, showPacmanPlot=True):
        print 'Training with gradient ...'

        if showPlot:
            # Initialize list to store loss per iteration for plotting later
            trainingLossPerIteration = []
            
            if showPacmanPlot:
                pacmanDisplay = pacmanPlot.PacmanPlotRegression();
                pacmanDisplay.plot(trainingData, regressionData)
                graphicsUtils.sleep(0.1)
            
        # Initializes weights to zero
        numDimensions = trainingData[0].size
        self.weights = np.zeros(numDimensions)
        
        # Stochastic gradient descent
        for i in xrange(numIterations):
            if i+1 % 10 == 0:
                print "Iteration " + str(i+1) + " of "+ str(numIterations)
                
            for (datum, label) in zip(trainingData, regressionData):
                self.weights = stochasticGradientDescentUpdate(datum, label, self.weights, self.alpha, self.der_loss_dw)

            if showPlot:
                trainingLoss = self.regressionLoss(trainingData, regressionData)
                trainingLossPerIteration.append(trainingLoss)
                
                if showPacmanPlot:
                    pacmanDisplay.setWeights(self.weights)
                    graphicsUtils.sleep(0.05)
                else:
                    plotUtil.plotRegression(trainingData,regressionData, self.weights, 1)
                    plotUtil.plotCurve(range(len(trainingLossPerIteration)), trainingLossPerIteration, 2, "Training Loss")
        if showPlot and showPacmanPlot:
            graphicsUtils.end_graphics()

    def regress(self, data):
        print 'Doing regression ...'

        numData = len(data)
        regressionResults = np.zeros(numData)

        # For each input x, predict y
        for (i, x) in enumerate(data):
            y = self.hypothesis([x, 1])
            regressionResults[i] = y

        return regressionResults
    
    def hypothesis(self, x):
        """
        Question 1
        
        Implement the linear regression hypothesis function. Given input array x, predict
        the scalar output value y, using the current value of self.weights.
        
        x is an array of the same length as self.weights (both include the bias term)
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()
        
    def regressionLoss(self, x_data, y_data):
        """
        Average loss across many data points
        """
        N = len(x_data)
        totalLoss = 0
        for (x, y) in zip(x_data, y_data):
            totalLoss += self.loss(x, y)
        return totalLoss/N
    
    def loss(self, x, y_true):
        """
        Question 1
        
        Quadratic loss comparing y_true to the hypothesis for a single data point x
        Returns a single float value for the loss
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()
        
    def der_loss_dw(self, x, y_true, weights):
        """
        Question 2
        
        Derivative of self.loss function with respect to self.weights, given a single data point x and
        label y_true.
        Returns an array of the derivative of the loss function with respect to each self.weights term
        [der_loss_dw1, der_loss_dw2, ...]
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

class BinaryLinearClassifier:
    """
    Simple linear classifier.
    """
    def __init__( self, legalLabels, max_iterations):
        self.type = "binaryLinear"
        assert len(legalLabels) == 2, "BinaryLinearClassifier requires number of legal labels to be exactly two"
        self.legalLabels = legalLabels
        self.alpha = 0.1 # Make sure this hard-coded value works (too large, things can blow up; too small, it takes forever)
        self.max_iterations = max_iterations

    def train( self, trainingData, trainingLabels, validationData, validationLabels, showPlot=True, showPacmanPlot=True):
        """
        Stochastic gradient descent to learn self.weights
        """
        numDimensions = trainingData[0].size
        
        # Initializes weights to zero
        self.weights = np.zeros(numDimensions)
        
        if showPlot:
            # Initialize list to store loss per iteration for plotting later
            trainingLossPerIteration = []
            # Initial loss
            trainingLoss = self.classificationLoss(trainingData, trainingLabels)
            trainingLossPerIteration.append(trainingLoss)

            # Check for offset term
            plotDims = numDimensions-1
            for datum in trainingData:
                if datum[-1] != 1:
                    plotDims += 1
                    break
                 
            if showPacmanPlot and plotDims <=2:
                if plotDims == 2:
                    pacmanDisplay = pacmanPlot.PacmanPlotClassification2D();
                    pacmanDisplay.plot(trainingData[:,:plotDims], trainingLabels)
                else:
                    pacmanDisplay = pacmanPlot.PacmanPlotLogisticRegression1D();
                    pacmanDisplay.plot(trainingData[:,0], trainingLabels)

                graphicsUtils.sleep(0.1)
            
        # Stochastic gradient descent
        for itr in xrange(self.max_iterations):
                
            for (datum, label) in zip(trainingData, trainingLabels):
                self.weights = stochasticGradientDescentUpdate(datum, label, self.weights, self.alpha, self.der_loss_dw)

            if showPlot:
                predictions = self.classify(validationData)
                accuracyCount = [predictions[i] == validationLabels[i] for i in range(len(validationLabels))].count(True)
                print "Performance on validation set for iteration= %d: (%.1f%%)" % (itr, 100.0*accuracyCount/len(validationLabels))

                trainingLoss = self.classificationLoss(trainingData, trainingLabels)
                trainingLossPerIteration.append(trainingLoss)
                
                if plotDims <= 2:
                    if showPacmanPlot:
                        pacmanDisplay.setWeights(self.weights)
                        graphicsUtils.sleep(0.1)
                    else:
                        if plotDims == 2:
                            plotUtil.plotClassification2D(trainingData[:,:plotDims],trainingLabels, self.weights, 1)
                        else:
                            plotUtil.plotLogisticRegression1D(trainingData[:,:plotDims],trainingLabels, self.weights, 1)
                plotUtil.plotCurve(range(len(trainingLossPerIteration)), trainingLossPerIteration, 2, "Training Loss")

        if showPlot and showPacmanPlot:
            graphicsUtils.end_graphics()


    def classify(self, data ):
        """
        Classifies each datum by rounding the output of the hypothesis function to either 0 or 1.
        """
        predicted_labels = []
        for x in data:
            y = self.hypothesis(x)
            predicted_label = self.legalLabels[int(round(y))]
            predicted_labels.append(predicted_label)
        return predicted_labels

    def hypothesis(self, x):
        """
        Question 3
        
        Implement the logistic regresssion hypothesis function (dot product
        of input and weights passed to a sigmoid function).
        In other words, given input array x and your current self.weights, return the 
        probability that x belongs to class 1 (rather than class 0).
        
        x: is an array of the same length as self.weights
        Returns a scalar between 0.0 and 1.0
        Note: No need to worry about a bias term. If one exists, it 
        has already been included in both x and self.weights.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

    def classificationLoss(self, x_data, y_data):
        """
        Average loss across many data points
        """
        N = len(x_data)
        totalLoss = 0
        for (x, y) in zip(x_data, y_data):
            totalLoss += self.loss(x, y)
        return totalLoss/N
    
    def loss(self, x, y_true):
        """
        Question 3
        
        Quadratic loss comparing label y_true to the hypothesis for a single data point x
        Returns a single float value for the loss
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

    def der_loss_dw(self, x, y_true, weights):
        """
        Question 3
        
        Derivative of self.loss function with respect to the input weights, given a single data point x and
        label y_true.
        Returns an array of the derivative of the loss function with respect to each input weights term
        [der_loss_dw1, der_loss_dw2, ...]
        
        Hint: There are three functions involved in the complete loss function (quadLoss, sigmoid, dot product).
        You have already implemented the derivatives for these three functions with respect to their inputs.
        You should be able to use the chain rule and these derivative functions.
        
        Another hint: To implement the hint above, you may need to first compute the input to each of the
        three functions involved in the complete loss function. For example, if you wanted to use der_sigmoid_dx,
        what value would you need to pass as input to that function?
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

class MulticlassLinearClassifier:
    """
    Simple multiclass linear classifier.
    """
    def __init__( self, legalLabels, max_iterations):
        self.type = "multiclassLinear"
        self.legalLabels = legalLabels
        self.labelIndex = {}
        for i in range(len(legalLabels)):
            self.labelIndex[legalLabels[i]] = i
        self.alpha = 0.01 # Make sure this hard-coded value works (too large, things can blow up; too small, it takes forever)
        self.max_iterations = max_iterations

    def train( self, trainingData, trainingLabels, validationData, validationLabels, showPlot=True, showPacmanPlot=True):
        """
        Stochastic gradient descent to learn self.weights
        """
        numDimensions = trainingData[0].size
        
        if showPlot:
            # Initialize list to store loss per iteration for plotting later
            trainingLossPerIteration = []
                        
        self.weights = []
        for i in xrange(len(self.legalLabels)):
            self.weights.append(np.zeros(len(trainingData[0])))
        
        # Stochastic gradient descent
        for itr in xrange(self.max_iterations):
                
            for (datum, label) in zip(trainingData, trainingLabels):
                # We have a list of arrays of weights here, instead of a matrix,
                # so we end up looping over labels
                dw = self.der_loss_dw(datum, label, self.weights)
                for j in range(len(self.legalLabels)):
                    self.weights[j] = self.weights[j] - self.alpha*dw[j]

            if showPlot:
                predictions = self.classify(validationData)
                accuracyCount = [predictions[i] == validationLabels[i] for i in range(len(validationLabels))].count(True)
                print "Performance on validation set for iteration= %d: (%.1f%%)" % (itr, 100.0*accuracyCount/len(validationLabels))

                trainingLoss = self.classificationLoss(trainingData, trainingLabels)
                trainingLossPerIteration.append(trainingLoss)
                
                plotUtil.plotCurve(range(len(trainingLossPerIteration)), trainingLossPerIteration, 2, "Training Loss")


    def classify(self, data ):
        """
        Classifies each datum as the label corresponding to the index of the maximum value in the
        hypothesis output array.
        """
        predicted_labels = []
        for x in data:
            y = self.hypothesis(x)
            bestLabelIndex = np.argmax(y)
            bestLabel = self.legalLabels[bestLabelIndex]
            predicted_labels.append(bestLabel)
        return predicted_labels

    def hypothesis(self, x):
        """
        Question 4
        
        Implement the softmax regresssion hypothesis function.
        Specifically, for each possible label, compute the dot product of the data and weights for 
        that label. Then we pass the output of each of those dot products into the softmax function.
        The function will return the output of that softmax function call, which is an array of
        values between 0.0 and 1.0 for each possible label.
        
        For this multiclass classification, self.weights is actually a list, where self.weights[i] 
        is the array of weights for the i-th possible label.
        
        x: is an array of the same length as each of the self.weights[i]
        Returns an array of values between 0.0 and 1.0; one value for each posible label
        Note: No need to worry about a bias term. If one exists, it 
        has already been included in both x and self.weights[i].
        """
        numClasses = len(self.legalLabels)

        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

    def classificationLoss(self, x_data, y_data):
        """
        Average loss across many data points
        """
        N = len(x_data)
        totalLoss = 0
        for (x, y) in zip(x_data, y_data):
            totalLoss += self.loss(x, y)
        return totalLoss/N
    
    def loss(self, x, y_true):
        """
        Cross entropy loss comparing label y_true to the hypothesis for a single data point x
        Returns a single float value for the loss
        """
        return crossEntLossLabel(self.hypothesis(x), y_true)

    def der_loss_dw(self, datum, label, weights):
        """
        Derivative of self.loss function with respect to the input weights, given a single data point x and
        label y_true.
        Returns the derivative of the loss function with respect to the input weights
        """
        numLabels = len(self.legalLabels)
        dot_output = np.zeros(numLabels)
        for i in xrange(numLabels):
            dot_output[i] = np.dot(datum, weights[i])
        softmax_output = softmax(dot_output)

        dloss_dw = []
        dloss_dsoftmax = der_crossEntLossLabel_dxlabel(softmax_output, label)
        for i in xrange(numLabels):
            dsoftmax_ddot = der_softmax_dx(dot_output, label, i)
            ddot_dw = der_dot_dw(datum, weights[i])

            dloss_dw.append(dloss_dsoftmax*dsoftmax_ddot*ddot_dw)
            if np.abs(np.max(dloss_dw[i])) > 100:
                print dloss_dw
        return dloss_dw

class OneVsRestLinearClassifier:
    """
    One-vs-rest multiclass classifier. Trains a BinaryLinearClassifier for each
    possible label. Then for classification it takes the max of a call to
    BinaryLinearClassifier.hypothesis for each possible label.
    """
    def __init__( self, legalLabels, max_iterations):
        self.type = "oneVsRestLinear"
        self.legalLabels = legalLabels
        self.alpha = 0.01 # Make sure this hard-coded value works, also if too large things can blow up
        self.max_iterations = max_iterations

    def train( self, trainingData, trainingLabels, validationData, validationLabels, showPlot=True ):
        """
        Trains a BinaryLinearClassifier (self.binaryClassifiers) for each possible label.
        """
        self.binaryClassifiers = []
        for legalLabel in self.legalLabels:
            print "Training class ", legalLabel
            oneVsRestTrainingLabels = [int(trainingLabel == legalLabel) for trainingLabel in trainingLabels]
            oneVsRestValidationLabels = [validationLabel == legalLabel for validationLabel in validationLabels]

            classifier = BinaryLinearClassifier([0,1], self.max_iterations)
            classifier.train(trainingData, oneVsRestTrainingLabels, validationData, oneVsRestValidationLabels, showPlot=showPlot)
            self.binaryClassifiers.append(classifier)

    def classify(self, data ):
        """
        Classify takes the max overs the values returned by a call to BinaryLinearClassifier.hypothesis
        for each possible label.
        """
        predicted_labels = []
        for datum in data:
            bestScore = float("-inf")
            bestLabelIndex = -1
            for i in range(len(self.legalLabels)):
                classificationScore = self.binaryClassifiers[i].hypothesis(datum)
                if classificationScore >= bestScore:
                    bestScore = classificationScore
                    bestLabelIndex = i

            bestLabel = self.legalLabels[bestLabelIndex]
            predicted_labels.append(bestLabel)
        return predicted_labels

