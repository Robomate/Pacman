# valueIterationAgents.py
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


import mdp, util,sys

from learningAgents import ValueEstimationAgent
import collections
import time

class AsynchronousValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = collections.defaultdict(float)
        states = self.mdp.getStates()
        for state in states:
            self.values[state] = 0

        "*** YOUR CODE HERE ***"
        for i in range(self.iterations):
            state = states[i % len(states)]
            if not self.mdp.isTerminal(state):
                action = self.computeActionFromValues(state)
                self.values[state] = self.computeQValueFromValues(state, action)

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        "util.raiseNotDefined()"
        qvalue = 0.0
        for nextstate, prob in self.mdp.getTransitionStatesAndProbs(state, action):
            Reward = self.mdp.getReward(state)
            qvalue += prob * (Reward + (self.discount * self.values[nextstate]))
        return qvalue

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        "util.raiseNotDefined()"
        Q = float('-inf')
        A = 0
        actions = self.mdp.getPossibleActions(state)
        if len(actions) == 0:
            return None
        for action in actions:
            if self.computeQValueFromValues(state, action) > Q:
                Q = self.computeQValueFromValues(state, action)
                A = action
        return A

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = collections.defaultdict(float)
        states = self.mdp.getStates()
        for state in states:
            self.values[state] = 0

        "*** YOUR CODE HERE ***"
        # Initialization
        predecessors = dict()
        priorityQueue = util.PriorityQueue()

        # Compute predecessors of all states
        for state in states:
            predecessorSet = set()
            if not self.mdp.isTerminal(state):
                for s in states:
                    if not self.mdp.isTerminal(s):
                        for direction in ['south', 'north', 'east', 'west']:
                            if direction in self.mdp.getPossibleActions(s):
                                for nextstate, prob in self.mdp.getTransitionStatesAndProbs(s, direction):
                                    if nextstate == state and prob > 0:
                                        predecessorSet.add(s)
            predecessors[state] =  predecessorSet
 
        # For each non-terminal state, do:
        for s in states:
            if not self.mdp.isTerminal(s):
                action = self.computeActionFromValues(s)
                highest = self.computeQValueFromValues(s, action)
                diff = abs(self.values[s] - highest) 
                priorityQueue.update(s, -diff) 

        # For iteration in 0, 1, 2, ..., self.iterations - 1, do:
        for i in range(self.iterations):
            if priorityQueue.isEmpty():
                return
            state = priorityQueue.pop()  
            if not self.mdp.isTerminal(state):
                action = self.computeActionFromValues(state)
                self.values[state] = self.computeQValueFromValues(state, action)
            for p in list(predecessors[state]):
                action = self.computeActionFromValues(p)
                highest = self.computeQValueFromValues(p, action)
                diff = abs(self.values[p] - highest) 
                if diff > theta:
                    priorityQueue.update(p, -diff) 

