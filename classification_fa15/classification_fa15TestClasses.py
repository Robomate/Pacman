# classification_fa15TestClasses.py
# ---------------------------------
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


from hashlib import sha1
import testClasses
# import json

from collections import defaultdict
from pprint import PrettyPrinter
pp = PrettyPrinter()

# from game import Agent
from pacman import GameState
from ghostAgents import RandomGhost
import random, math, traceback, sys, os
import time
import layout, pacman
# import autograder
# import grading

import dataClassifier, samples
from featureExtractors import EnhancedExtractor
from featureExtractors import SimpleExtractor
import numpy as np
from util import FixedRandom

VERBOSE = False

import gridworld



# Data sets
# ---------

EVAL_MULTIPLE_CHOICE=True

numTraining = 100
TEST_SET_SIZE = 100
DIGIT_DATUM_WIDTH=28
DIGIT_DATUM_HEIGHT=28

def readDigitData(trainingSize=100, testSize=100):
    rootdata = 'digitdata/'
    # loading digits data
    rawTrainingData = samples.loadDataFile(rootdata + 'trainingimages', trainingSize,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
    trainingLabels = samples.loadLabelsFile(rootdata + "traininglabels", trainingSize)
    rawValidationData = samples.loadDataFile(rootdata + "validationimages", TEST_SET_SIZE,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
    validationLabels = samples.loadLabelsFile(rootdata + "validationlabels", TEST_SET_SIZE)
    rawTestData = samples.loadDataFile("digitdata/testimages", testSize,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
    testLabels = samples.loadLabelsFile("digitdata/testlabels", testSize)
    try:
        print "Extracting features..."
        featureFunction = dataClassifier.basicFeatureExtractorDigit
        trainingData = map(featureFunction, rawTrainingData)
        validationData = map(featureFunction, rawValidationData)
        testData = map(featureFunction, rawTestData)
    except:
        display("An exception was raised while extracting basic features: \n %s" % getExceptionTraceBack())
    return (trainingData, trainingLabels, validationData, validationLabels, rawTrainingData, rawValidationData, testData, testLabels, rawTestData)

def readSuicideData(trainingSize=100, testSize=100):
    rootdata = 'pacmandata'
    rawTrainingData, trainingLabels = samples.loadPacmanData(rootdata + '/suicide_training.pkl', trainingSize)
    rawValidationData, validationLabels = samples.loadPacmanData(rootdata + '/suicide_validation.pkl', testSize)
    rawTestData, testLabels = samples.loadPacmanData(rootdata + '/suicide_test.pkl', testSize)
    trainingData = []
    validationData = []
    testData = []
    return (trainingData, trainingLabels, validationData, validationLabels, rawTrainingData, rawValidationData, testData, testLabels, rawTestData)

def readContestData(trainingSize=100, testSize=100):
    rootdata = 'pacmandata'
    rawTrainingData, trainingLabels = samples.loadPacmanData(rootdata + '/contest_training.pkl', trainingSize)
    rawValidationData, validationLabels = samples.loadPacmanData(rootdata + '/contest_validation.pkl', testSize)
    rawTestData, testLabels = samples.loadPacmanData(rootdata + '/contest_test.pkl', testSize)
    trainingData = []
    validationData = []
    testData = []
    return (trainingData, trainingLabels, validationData, validationLabels, rawTrainingData, rawValidationData, testData, testLabels, rawTestData)

def simple1D():
    X = np.array([-10, -5, -1, 1, 5, 10])+3
    Y = np.array([1, 1, 1, 0, 0, 0])
    vX = np.array([-9, -4, 4, 9])+3
    vY = np.array([1, 1, 0, 0])
    tX = np.array([-7, 7])+3
    tY = np.array([1, 0])
    
    X = np.hstack((X[:,None], np.ones( (X.size,1) )))
    vX = np.hstack((vX[:,None], np.ones( (vX.size,1) )))
    tX = np.hstack((tX[:,None], np.ones( (tX.size,1) )))
    
    return (X, Y, vX, vY, X, vX, tX, tY, tX)

def simple2D():
    X = np.array([[-.2, -.2],[-.5, 0],[-.05, 0],[-.1, 0],
                  [0, .4], [.5, .5], [1, 0], [0.5, 0]]) * 10
    x_bar = np.mean(X) - 3
    X = X - x_bar
    Y = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    vX = np.array([[-.6, -.6], [0.1, 0], [0, -0.3],
                   [.6, .6], [0.3, 0], [0, 0.3]]) * 10
    vX = vX - x_bar
    vY = np.array([1, 1, 1, 0, 0, 0])
    tX = np.array([[-.1, 0], [.4, 0], [-.2, 0], [.2, .3]]) * 10
    tX = tX - x_bar
    tY = np.array([1, 0, 1, 0])

    X = np.hstack((X, np.ones( (X.shape[0],1) )))
    vX = np.hstack((vX, np.ones( (vX.shape[0],1) )))
    tX = np.hstack((tX, np.ones( (tX.shape[0],1) )))    

    return (X, Y, vX, vY, X, vX, tX, tY, tX)

def GMM2D():
    np.random.seed(1)
    mu1 = np.array([-1, 1])
    mu2 = np.array([2, -2])
    X_1 = np.random.normal(mu1, size=(100, 2))*2
    X_2 = np.random.normal(mu2, size=(100, 2))*2
    X = np.r_[X_1[:50], X_2[:50]]
    X = np.c_[X, np.ones((100,1))]
    Y = np.ones((100))
    Y[50:] = 0

    vX = np.r_[X_1[50:80], X_2[50:80]]
    vX = np.c_[vX, np.ones((60,1))]
    vY = np.ones((60))
    vY[30:] = 0

    tX = np.r_[X_1[80:], X_2[80:]]
    tX = np.c_[tX, np.ones((40,1))]
    tY = np.ones((40))
    tY[20:] = 0
    return (X, Y, vX, vY, X, vX, tX, tY, tX)



def tinyDataSet():
    def count(m,b,h):
        c = util.Counter();
        c['m'] = m;
        c['b'] = b;
        c['h'] = h;
        return c;

    training = [count(0,0,0), count(1,0,0), count(1,1,0), count(0,1,1), count(1,0,1), count(1,1,1)]
    trainingLabels = [1,        1,            1           , 1           ,      -1     ,      -1]

    validation = [count(1,0,1)]
    validationLabels = [ 1]

    test = [count(1,0,1)]
    testLabels = [-1]

    return (training,trainingLabels,validation,validationLabels,test,testLabels);


def tinyDataSetPeceptronAndMira():
    def count(m,b,h):
        c = util.Counter();
        c['m'] = m;
        c['b'] = b;
        c['h'] = h;
        return c;

    training = [count(1,0,0), count(1,1,0), count(0,1,1), count(1,0,1), count(1,1,1)]
    trainingLabels = [1,            1,            1,          -1      ,      -1]

    validation = [count(1,0,1)]
    validationLabels = [ 1]

    test = [count(1,0,1)]
    testLabels = [-1]

    return (training,trainingLabels,validation,validationLabels,test,testLabels);

def parseGrid(string):
    grid = [[entry.strip() for entry in line.split()] for line in string.split('\n')]
    for row in grid:
        for x, col in enumerate(row):
            try:
                col = int(col)
            except:
                pass
            if col == "_":
                col = ' '
            row[x] = col
    return gridworld.makeGrid(grid)


# Test classes
# ------------
class ApproximateQLearningTest(testClasses.TestCase):

    def __init__(self, question, testDict):
        super(ApproximateQLearningTest, self).__init__(question, testDict)
        self.discount = float(testDict['discount'])
        self.grid = gridworld.Gridworld(parseGrid(testDict['grid']))
        if 'noise' in testDict: self.grid.setNoise(float(testDict['noise']))
        if 'livingReward' in testDict: self.grid.setLivingReward(float(testDict['livingReward']))
        self.grid = gridworld.Gridworld(parseGrid(testDict['grid']))
        self.env = gridworld.GridworldEnvironment(self.grid)
        self.epsilon = float(testDict['epsilon'])
        self.learningRate = float(testDict['learningRate'])
        self.extractor = 'IdentityExtractor'
        if 'extractor' in testDict:
            self.extractor = testDict['extractor']
        self.opts = {'actionFn': self.env.getPossibleActions, 'epsilon': self.epsilon, 'gamma': self.discount, 'alpha': self.learningRate}
        numExperiences = int(testDict['numExperiences'])
        maxPreExperiences = 10
        self.numsExperiencesForDisplay = range(min(numExperiences, maxPreExperiences))
        self.testOutFile = testDict['test_out_file']
        if maxPreExperiences < numExperiences:
            self.numsExperiencesForDisplay.append(numExperiences)

    def writeFailureFile(self, string):
        with open(self.testOutFile, 'w') as handle:
            handle.write(string)

    def removeFailureFileIfExists(self):
        if os.path.exists(self.testOutFile):
            os.remove(self.testOutFile)

    def execute(self, grades, moduleDict, solutionDict):
        failureOutputFileString = ''
        failureOutputStdString = ''
        for n in self.numsExperiencesForDisplay:
            testPass, stdOutString, fileOutString = self.executeNExperiences(grades, moduleDict, solutionDict, n)
            failureOutputStdString += stdOutString
            failureOutputFileString += fileOutString
            if not testPass:
                self.addMessage(failureOutputStdString)
                self.addMessage('For more details to help you debug, see test output file %s\n\n' % self.testOutFile)
                self.writeFailureFile(failureOutputFileString)
                return self.testFail(grades)
        self.removeFailureFileIfExists()
        return self.testPass(grades)

    def executeNExperiences(self, grades, moduleDict, solutionDict, n):
        testPass = True
        qValuesPretty, weights, actions, lastExperience = self.runAgent(moduleDict, n)
        stdOutString = ''
        fileOutString = "==================== Iteration %d ====================\n" % n
        if lastExperience is not None:
            fileOutString += "Agent observed the transition (startState = %s, action = %s, endState = %s, reward = %f)\n\n" % lastExperience
        weightsKey = 'weights_k_%d' % n
        if weights == eval(solutionDict[weightsKey]):
            fileOutString += "Weights at iteration %d are correct." % n
            fileOutString += "   Student/correct solution:\n\n%s\n\n" % pp.pformat(weights)
        for action in actions:
            qValuesKey = 'q_values_k_%d_action_%s' % (n, action)
            qValues = qValuesPretty[action]
            if self.comparePrettyValues(qValues, solutionDict[qValuesKey]):
                fileOutString += "Q-Values at iteration %d for action '%s' are correct." % (n, action)
                fileOutString += "   Student/correct solution:\n\t%s" % self.prettyValueSolutionString(qValuesKey, qValues)
            else:
                testPass = False
                outString = "Q-Values at iteration %d for action '%s' are NOT correct." % (n, action)
                outString += "   Student solution:\n\t%s" % self.prettyValueSolutionString(qValuesKey, qValues)
                outString += "   Correct solution:\n\t%s" % self.prettyValueSolutionString(qValuesKey, solutionDict[qValuesKey])
                stdOutString += outString
                fileOutString += outString
        return testPass, stdOutString, fileOutString

    def writeSolution(self, moduleDict, filePath):
        with open(filePath, 'w') as handle:
            for n in self.numsExperiencesForDisplay:
                qValuesPretty, weights, actions, _ = self.runAgent(moduleDict, n)
                handle.write(self.prettyValueSolutionString('weights_k_%d' % n, pp.pformat(weights)))
                for action in actions:
                    handle.write(self.prettyValueSolutionString('q_values_k_%d_action_%s' % (n, action), qValuesPretty[action]))
        return True

    def runAgent(self, moduleDict, numExperiences):
        agent = moduleDict['qlearningAgents'].ApproximateQAgent(extractor=self.extractor, **self.opts)
        states = filter(lambda state : len(self.grid.getPossibleActions(state)) > 0, self.grid.getStates())
        states.sort()
        randObj = FixedRandom().random
        # choose a random start state and a random possible action from that state
        # get the next state and reward from the transition function
        lastExperience = None
        for i in range(numExperiences):
            startState = randObj.choice(states)
            action = randObj.choice(self.grid.getPossibleActions(startState))
            (endState, reward) = self.env.getRandomNextState(startState, action, randObj=randObj)
            lastExperience = (startState, action, endState, reward)
            agent.update(*lastExperience)
        actions = list(reduce(lambda a, b: set(a).union(b), [self.grid.getPossibleActions(state) for state in states]))
        qValues = {}
        weights = agent.getWeights()
        for state in states:
            possibleActions = self.grid.getPossibleActions(state)
            for action in actions:
                if not qValues.has_key(action):
                    qValues[action] = {}
                if action in possibleActions:
                    qValues[action][state] = agent.getQValue(state, action)
                else:
                    qValues[action][state] = None
        qValuesPretty = {}
        for action in actions:
            qValuesPretty[action] = self.prettyValues(qValues[action])
        return (qValuesPretty, weights, actions, lastExperience)

    def prettyPrint(self, elements, formatString):
        pretty = ''
        states = self.grid.getStates()
        for ybar in range(self.grid.grid.height):
            y = self.grid.grid.height-1-ybar
            row = []
            for x in range(self.grid.grid.width):
                if (x, y) in states:
                    value = elements[(x, y)]
                    if value is None:
                        row.append('   illegal')
                    else:
                        row.append(formatString.format(elements[(x,y)]))
                else:
                    row.append('_' * 10)
            pretty += '        %s\n' % ("   ".join(row), )
        pretty += '\n'
        return pretty

    def prettyValues(self, values):
        return self.prettyPrint(values, '{0:10.4f}')

    def prettyPolicy(self, policy):
        return self.prettyPrint(policy, '{0:10s}')

    def prettyValueSolutionString(self, name, pretty):
        return '%s: """\n%s\n"""\n\n' % (name, pretty.rstrip())

    def comparePrettyValues(self, aPretty, bPretty, tolerance=0.01):
        aList = self.parsePrettyValues(aPretty)
        bList = self.parsePrettyValues(bPretty)
        if len(aList) != len(bList):
            return False
        for a, b in zip(aList, bList):
            try:
                aNum = float(a)
                bNum = float(b)
                # error = abs((aNum - bNum) / ((aNum + bNum) / 2.0))
                error = abs(aNum - bNum)
                if error > tolerance:
                    return False
            except ValueError:
                if a.strip() != b.strip():
                    return False
        return True

    def parsePrettyValues(self, pretty):
        values = pretty.split()
        return values

def getAccuracy(data, classifier, featureFunction=dataClassifier.basicFeatureExtractorDigit,
                showPlot=True):
    trainingData, trainingLabels, validationData, validationLabels, rawTrainingData, rawValidationData, testData, testLabels, rawTestData = data
    if featureFunction != dataClassifier.basicFeatureExtractorDigit:
        trainingData = map(featureFunction, rawTrainingData)
        validationData = map(featureFunction, rawValidationData)
        testData = map(featureFunction, rawTestData)
    classifier.train(trainingData, trainingLabels, validationData, validationLabels,
                     showPlot=showPlot)
    guesses = classifier.classify(testData)
    correct = [guesses[i] == testLabels[i] for i in range(len(testLabels))].count(True)
    acc = 100.0 * correct / len(testLabels)
    serialized_guesses = ", ".join([str(guesses[i]) for i in range(len(testLabels))])
    print str(correct), ("correct out of " + str(len(testLabels)) + " (%.1f%%).") % (acc)
    return acc, serialized_guesses

class GradeRegressorTest(testClasses.TestCase):

    def __init__(self, question, testDict):
        super(GradeRegressorTest, self).__init__(question, testDict)
        self.showPlot = not question.display.checkNullDisplay()

        self.fname = testDict['fname']
        self.numIter = int(testDict['numIter'])
        data = np.load(self.fname)
        self.X = data["data"]
        self.X = np.hstack((self.X[:, None], np.ones((self.X.size, 1))))
        self.Y = data["regressionResults"]

        self.maxPoints = int(testDict['maxPoints'])

        self.tolerance = float(testDict['tolerance'])

    def execute(self, grades, moduleDict, solutionDict):
        regressor = moduleDict['linearLearning'].LinearRegression()
        totalPoints = 0
        regressor.trainGradient(self.X, self.Y, self.numIter, showPlot=self.showPlot)
        sol_weights = solutionDict['weights'].strip('[').strip(']').split()
        sol_weights = np.array([float(w) for w in sol_weights])
        if np.allclose(sol_weights, regressor.weights, atol=self.tolerance):
            totalPoints = self.maxPoints
        else:
            print "Regression Test Failed"
            print "Student weights:\t", regressor.weights
            print "Solution weights:\t", sol_weights
        return self.testPartial(grades, totalPoints, self.maxPoints)

    def writeSolution(self, moduleDict, filePath):
        regressor = moduleDict['linearLearning'].LinearRegression()
        handle = open(filePath, 'w')
        handle.write('# This is the solution file for %s.\n' % self.path)
        regressor.trainGradient(self.X, self.Y, self.numIter, showPlot=self.showPlot)
        handle.write('weights: "{}"\n'.format(regressor.weights))
        handle.close()
        return True


class GradeClassifierTest(testClasses.TestCase):

    def __init__(self, question, testDict):
        super(GradeClassifierTest, self).__init__(question, testDict)


        self.classifierModule = testDict['classifierModule']
        self.classifierClass = testDict['classifierClass']
        self.datasetName = testDict['datasetName']

        self.accuracyScale = int(testDict['accuracyScale'])
        self.accuracyThresholds = [int(s) for s in testDict.get('accuracyThresholds','').split()]
        self.exactOutput = testDict['exactOutput'].lower() == "true"

        self.automaticTuning = testDict['automaticTuning'].lower() == "true" if 'automaticTuning' in testDict else None
        self.max_iterations = int(testDict['max_iterations']) if 'max_iterations' in testDict else None
        self.featureFunction = testDict['featureFunction'] if 'featureFunction' in testDict else 'basicFeatureExtractorDigit'

        self.maxPoints = len(self.accuracyThresholds) * self.accuracyScale
        if "extraPoints" in testDict.keys():
            self.maxPoints = self.maxPoints - int(testDict["extraPoints"])

        self.showPlot = not question.display.checkNullDisplay()


    def grade_classifier(self, moduleDict):
        smallDigitData = readDigitData(20)
        bigDigitData = readDigitData(1000)

        suicideData = readSuicideData(1000)
        contestData = readContestData(1000)

        DATASETS = {
            "smallDigitData": lambda: smallDigitData,
            "bigDigitData": lambda: bigDigitData,
            "simple1D": simple1D,
            "simple2D": simple2D,
            "gmm2D": GMM2D,
            "tinyDataSet": tinyDataSet,
            "tinyDataSetPeceptronAndMira": tinyDataSetPeceptronAndMira,
            "suicideData": lambda: suicideData,
            "contestData": lambda: contestData
        }

        DATASETS_LEGAL_LABELS = {
            "smallDigitData": range(10),
            "bigDigitData": range(10),
            "simple1D": [0, 1],
            "simple2D": [0, 1],
            "gmm2D": [0, 1],
            "tinyDataSet": [-1,1],
            "tinyDataSetPeceptronAndMira": [-1,1],
            "suicideData": ["EAST", 'WEST', 'NORTH', 'SOUTH', 'STOP'],
            "contestData": ["EAST", 'WEST', 'NORTH', 'SOUTH', 'STOP']
        }

        featureFunction = getattr(dataClassifier, self.featureFunction)
        data = DATASETS[self.datasetName]()
        legalLabels = DATASETS_LEGAL_LABELS[self.datasetName]
        try:
            classifierClass = getattr(moduleDict[self.classifierModule], self.classifierClass)
        except KeyError:
            import pdb; pdb.set_trace()

        if self.max_iterations != None:
            classifier = classifierClass(legalLabels, self.max_iterations)
        else:
            classifier = classifierClass(legalLabels)

        if self.automaticTuning != None:
            classifier.automaticTuning = self.automaticTuning

        return getAccuracy(data, classifier, featureFunction=featureFunction,
                           showPlot=self.showPlot)


    def execute(self, grades, moduleDict, solutionDict):
        accuracy, guesses = self.grade_classifier(moduleDict)

        # Either grade them on the accuracy of their classifer,
        # or their exact
        if self.exactOutput:
            gold_guesses = solutionDict['guesses']
            if guesses == gold_guesses:
                totalPoints = self.maxPoints
            else:
                self.addMessage("Incorrect classification after training:")
                self.addMessage("  student classifications: " + guesses)
                self.addMessage("  correct classifications: " + gold_guesses)
                totalPoints = 0
        else:
            # Grade accuracy
            totalPoints = 0
            for threshold in self.accuracyThresholds:
                if accuracy >= threshold:
                    totalPoints += self.accuracyScale

            # Print grading schedule
            self.addMessage("%s correct (%s of %s points)" % (accuracy, totalPoints, self.maxPoints))
            self.addMessage("    Grading scheme:")
            self.addMessage("     < %s:  0 points" % (self.accuracyThresholds[0],))
            for idx, threshold in enumerate(self.accuracyThresholds):
                self.addMessage("    >= %s:  %s points" % (threshold, (idx+1)*self.accuracyScale))

        return self.testPartial(grades, totalPoints, self.maxPoints)

    def writeSolution(self, moduleDict, filePath):
        handle = open(filePath, 'w')
        handle.write('# This is the solution file for %s.\n' % self.path)

        if self.exactOutput:
            _, guesses = self.grade_classifier(moduleDict)
            handle.write('guesses: "%s"' % (guesses,))

        handle.close()
        return True




class MultipleChoiceTest(testClasses.TestCase):

    def __init__(self, question, testDict):
        super(MultipleChoiceTest, self).__init__(question, testDict)
        self.ans = testDict['result']
        self.question = testDict['question']

    def execute(self, grades, moduleDict, solutionDict):
        studentSolution = str(getattr(moduleDict['answers'], self.question)())
        encryptedSolution = sha1(studentSolution.strip().lower()).hexdigest()
        if encryptedSolution == self.ans:
            return self.testPass(grades)
        else:
            self.addMessage("Solution is not correct.")
            self.addMessage("Student solution: %s" % studentSolution)
            return self.testFail(grades)

    def writeSolution(self, moduleDict, filePath):
        handle = open(filePath, 'w')
        handle.write('# This is the solution file for %s.\n' % self.path)
        handle.write('# File intentionally blank.\n')
        handle.close()
        return True


class UnitTest(testClasses.TestCase):

    def __init__(self, question, testDict):
        super(UnitTest, self).__init__(question, testDict)
        self.preamble = compile(testDict.get('preamble', ""), "%s.preamble" % self.getPath(), 'exec')
        self.test = compile(testDict['test'], "%s.test" % self.getPath(), 'eval')
        self.success = testDict['success']
        self.failure = testDict['failure']
        self.tolerance = float(testDict['tolerance'])
        self.partialPoints = 0
        if "partialPoints" in testDict.keys():
            self.partialPoints = int(testDict["partialPoints"])

    def evalCode(self, moduleDict):
        bindings = dict(moduleDict)
        exec self.preamble in bindings
        return eval(self.test, bindings)

    def execute(self, grades, moduleDict, solutionDict):
        result = self.evalCode(moduleDict)
        try:
            solution = float(solutionDict['result'])
        except ValueError:
            solution = solutionDict['result']
            solution = solution.replace('[','')
            solution = solution.replace(']','')
            solution = solution.split(' ')
            solution = [s for s in solution if s!='']
            for i in range(len(solution)):
                solution[i] = float(solution[i])
            solution = np.array(solution)

        error = result - solution
        errorNorm = np.linalg.norm(np.array(error))
        if errorNorm < self.tolerance:
            grades.addMessage('PASS: %s' % self.path)
            grades.addMessage('\t%s' % self.success)
            if self.partialPoints > 0:
                print "                    (%i of %i points)" % (self.partialPoints, self.partialPoints)
                grades.addPoints(self.partialPoints)
            return True
        else:
            grades.addMessage('FAIL: %s' % self.path)
            grades.addMessage('\t%s' % self.failure)
            grades.addMessage('\tstudent result: "%s"' % result)
            grades.addMessage('\tcorrect result: "%s"' % solutionDict['result'])
        if self.partialPoints > 0:
            print "                    (%i of %i points)" % (0, self.partialPoints)
        return False

    def writeSolution(self, moduleDict, filePath):
        handle = open(filePath, 'w')
        handle.write('# This is the solution file for %s.\n' % self.path)
        handle.write('# The result of evaluating the test must equal the below when cast to a string.\n')

        output = self.evalCode(moduleDict)
        # handle.write('result: "%f"\n' % output)
        # print '>>>>>>>>>>>>>>>>>>result:"%s"\n' % output
        handle.write('result:"%s"\n' % output)
        handle.close()
        return True


class EvalAgentTest(testClasses.TestCase):

    def __init__(self, question, testDict):
        super(EvalAgentTest, self).__init__(question, testDict)
        self.layoutName = testDict['layoutName']
        self.agentName = testDict['agentName']
        self.ghosts = eval(testDict['ghosts'])
        self.maxTime = int(testDict['maxTime'])
        self.seed = int(testDict['randomSeed'])
        self.numGames = int(testDict['numGames'])
        self.numTraining = int(testDict['numTraining'])

        self.scoreMinimum = int(testDict['scoreMinimum']) if 'scoreMinimum' in testDict else None
        self.nonTimeoutMinimum = int(testDict['nonTimeoutMinimum']) if 'nonTimeoutMinimum' in testDict else None
        self.winsMinimum = int(testDict['winsMinimum']) if 'winsMinimum' in testDict else None

        self.scoreThresholds = [int(s) for s in testDict.get('scoreThresholds','').split()]
        self.nonTimeoutThresholds = [int(s) for s in testDict.get('nonTimeoutThresholds','').split()]
        self.winsThresholds = [int(s) for s in testDict.get('winsThresholds','').split()]

        self.maxPoints = sum([len(t) for t in [self.scoreThresholds, self.nonTimeoutThresholds, self.winsThresholds]])
        self.agentArgs = testDict.get('agentArgs', '')

    def execute(self, grades, moduleDict, solutionDict):
        startTime = time.time()

        agentType = getattr(moduleDict['qlearningAgents'], self.agentName)
        agentOpts = pacman.parseAgentArgs(self.agentArgs) if self.agentArgs != '' else {}
        agent = agentType(**agentOpts)

        lay = layout.getLayout(self.layoutName, 3)

        disp = self.question.getDisplay()

        random.seed(self.seed)
        games = pacman.runGames(lay, agent, self.ghosts, disp, self.numGames, False, numTraining=self.numTraining, catchExceptions=True, timeout=self.maxTime)
        totalTime = time.time() - startTime

        stats = {'time': totalTime, 'wins': [g.state.isWin() for g in games].count(True),
                 'games': games, 'scores': [g.state.getScore() for g in games],
                 'timeouts': [g.agentTimeout for g in games].count(True), 'crashes': [g.agentCrashed for g in games].count(True)}

        averageScore = sum(stats['scores']) / float(len(stats['scores']))
        nonTimeouts = self.numGames - stats['timeouts']
        wins = stats['wins']

        def gradeThreshold(value, minimum, thresholds, name):
            points = 0
            passed = (minimum == None) or (value >= minimum)
            if passed:
                for t in thresholds:
                    if value >= t:
                        points += 1
            return (passed, points, value, minimum, thresholds, name)

        results = [gradeThreshold(averageScore, self.scoreMinimum, self.scoreThresholds, "average score"),
                   gradeThreshold(nonTimeouts, self.nonTimeoutMinimum, self.nonTimeoutThresholds, "games not timed out"),
                   gradeThreshold(wins, self.winsMinimum, self.winsThresholds, "wins")]

        totalPoints = 0
        for passed, points, value, minimum, thresholds, name in results:
            if minimum == None and len(thresholds)==0:
                continue

            # print passed, points, value, minimum, thresholds, name
            totalPoints += points
            if not passed:
                assert points == 0
                self.addMessage("%s %s (fail: below minimum value %s)" % (value, name, minimum))
            else:
                self.addMessage("%s %s (%s of %s points)" % (value, name, points, len(thresholds)))

            if minimum != None:
                self.addMessage("    Grading scheme:")
                self.addMessage("     < %s:  fail" % (minimum,))
                if len(thresholds)==0 or minimum != thresholds[0]:
                    self.addMessage("    >= %s:  0 points" % (minimum,))
                for idx, threshold in enumerate(thresholds):
                    self.addMessage("    >= %s:  %s points" % (threshold, idx+1))
            elif len(thresholds) > 0:
                self.addMessage("    Grading scheme:")
                self.addMessage("     < %s:  0 points" % (thresholds[0],))
                for idx, threshold in enumerate(thresholds):
                    self.addMessage("    >= %s:  %s points" % (threshold, idx+1))

        if any([not passed for passed, _, _, _, _, _ in results]):
            totalPoints = 0

        return self.testPartial(grades, totalPoints, self.maxPoints)

    def writeSolution(self, moduleDict, filePath):
        handle = open(filePath, 'w')
        handle.write('# This is the solution file for %s.\n' % self.path)
        handle.write('# File intentionally blank.\n')
        handle.close()
        return True

