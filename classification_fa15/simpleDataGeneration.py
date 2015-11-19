# simpleDataGeneration.py
# -----------------------
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


# This file is for generating random samples from the following linear Gaussian
# model: y = ax + b + G(m,var)

import random
import numpy as np

a = 1.77
b = 0.1
m = 0.2
sigma = 0.05

print "Generating data from: y = " +str(a)+"x+"+str(b)+"+" + "G(" + str(m) + ', '+str(sigma)+")..."

dataNum = 50
validationNum = 20
testNum = 50
dataRange = [-5,5]
datasetName = "simple50"

print "dataNum: " + str(dataNum)
print "validationNum: " + str(validationNum)
print "testNum: " + str(testNum)
print "dataRange: " + str(dataRange)
print "datasetName: " + datasetName + ".npz"



x = [random.random()*(dataRange[1]-dataRange[0])+dataRange[0] for i in xrange(dataNum)]
y = [a*s + b + random.gauss(m,sigma) for s in x]

xValidation = [random.random()*(dataRange[1]-dataRange[0])+dataRange[0] for i in xrange(validationNum)]
yValidation = [a*s + b + random.gauss(m,sigma) for s in xValidation]

xTest = [random.random()*(dataRange[1]-dataRange[0])+dataRange[0] for i in xrange(testNum)]
yTest = [a*s + b + random.gauss(m,sigma) for s in xTest]

np.savez(datasetName, data = x, regressionResults = y, paras = [a, b+m], \
         dataVali = xValidation, regressionResultsVali = yValidation, \
         dataTest = xTest, regressionResultsTest = yTest)
