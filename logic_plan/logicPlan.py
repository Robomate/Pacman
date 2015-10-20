# logicPlan.py
# ------------
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


"""
In logicPlan.py, you will implement logic planning methods which are called by
Pacman agents (in logicAgents.py).
"""

import util
import sys
import logic
import game


pacman_str = 'P'
ghost_pos_str = 'G'
ghost_east_str = 'GE'
pacman_alive_str = 'PA'

class PlanningProblem:
    """
    This class outlines the structure of a planning problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the planning problem.
        """
        util.raiseNotDefined()

    def getGhostStartStates(self):
        """
        Returns a list containing the start state for each ghost.
        Only used in problems that use ghosts (FoodGhostPlanningProblem)
        """
        util.raiseNotDefined()
        
    def getGoalState(self):
        """
        Returns goal state for problem. Note only defined for problems that have
        a unique goal state such as PositionPlanningProblem
        """
        util.raiseNotDefined()

def tinyMazePlan(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def sentence1():
    """Returns a logic.Expr instance that encodes that the following expressions are all true.
    
    A or B
    (not A) if and only if ((not B) or C)
    (not A) or (not B) or C
    """
    "*** YOUR CODE HERE ***"
    A, B, C = logic.Expr('A'), logic.Expr('B'), logic.Expr('C')
    line1, line2, line3 = A | B, ~A % (~B | C), logic.disjoin(~A, ~B, C)
    return logic.conjoin(line1, line2, line3)

def sentence2():
    """Returns a logic.Expr instance that encodes that the following expressions are all true.
    
    C if and only if (B or D)
    A implies ((not B) and (not D))
    (not (B and (not C))) implies A
    (not D) implies C
    """
    "*** YOUR CODE HERE ***"
    A, B, C, D = logic.Expr('A'), logic.Expr('B'), logic.Expr('C'), logic.Expr('D')
    line1, line2, line3, line4 = C % (B | D), A >> (~B & ~D), ~(B & ~C) >> A, ~D >> C
    return logic.conjoin(line1, line2, line3, line4)

def sentence3():
    """Using the symbols WumpusAlive[1], WumpusAlive[0], WumpusBorn[0], and WumpusKilled[0],
    created using the logic.PropSymbolExpr constructor, return a logic.PropSymbolExpr
    instance that encodes the following English sentences (in this order):

    The Wumpus is alive at time 1 if and only if the Wumpus was alive at time 0 and it was
    not killed at time 0 or it was not alive and time 0 and it was born at time 0.

    The Wumpus cannot both be alive at time 0 and be born at time 0.

    The Wumpus is born at time 0.
    """
    "*** YOUR CODE HERE ***"
    line1 = logic.PropSymbolExpr("WumpusAlive", 1) % ((logic.PropSymbolExpr("WumpusAlive", 0) & ~logic.PropSymbolExpr("WumpusKilled", 0)) | (~logic.PropSymbolExpr("WumpusAlive", 0) & logic.PropSymbolExpr("WumpusBorn", 0)))
    line2 = ~(logic.PropSymbolExpr("WumpusAlive", 0) & logic.PropSymbolExpr("WumpusBorn", 0))
    line3 = logic.PropSymbolExpr("WumpusBorn", 0)
    return logic.conjoin(line1, line2, line3)

def findModel(sentence):
    """Given a propositional logic sentence (i.e. a logic.Expr instance), returns a satisfying
    model if one exists. Otherwise, returns False.
    """
    "*** YOUR CODE HERE ***"
    return logic.pycoSAT(logic.to_cnf(sentence))

def atLeastOne(literals) :
    """
    Given a list of logic.Expr literals (i.e. in the form A or ~A), return a single 
    logic.Expr instance in CNF (conjunctive normal form) that represents the logic 
    that at least one of the literals in the list is true.
    >>> A = logic.PropSymbolExpr('A');
    >>> B = logic.PropSymbolExpr('B');
    >>> symbols = [A, B]
    >>> atleast1 = atLeastOne(symbols)
    >>> model1 = {A:False, B:False}
    >>> print logic.pl_true(atleast1,model1)
    False
    >>> model2 = {A:False, B:True}
    >>> print logic.pl_true(atleast1,model2)
    True
    >>> model3 = {A:True, B:True}
    >>> print logic.pl_true(atleast1,model2)
    True
    """
    "*** YOUR CODE HERE ***"
    return logic.disjoin(literals)    
         

def atMostOne(literals) :
    """
    Given a list of logic.Expr literals, return a single logic.Expr instance in 
    CNF (conjunctive normal form) that represents the logic that at most one of 
    the expressions in the list is true.
    """
    "*** YOUR CODE HERE ***"
    lst = []
    for i in range(len(literals)):
        for j in range(len(literals)):
            if i != j:
                l1, l2 = ~literals[i] | ~literals[j], ~literals[j] | ~literals[i] 
                if l1 not in lst and l2 not in lst:
                    lst.append(l1)
    return logic.conjoin(lst)


def exactlyOne(literals) :
    """
    Given a list of logic.Expr literals, return a single logic.Expr instance in 
    CNF (conjunctive normal form)that represents the logic that exactly one of 
    the expressions in the list is true.
    """
    "*** YOUR CODE HERE ***"
    return logic.conjoin(atLeastOne(literals), atMostOne(literals))


def extractActionSequence(model, actions):
    """
    Convert a model in to an ordered list of actions.
    model: Propositional logic model stored as a dictionary with keys being
    the symbol strings and values being Boolean: True or False
    Example:
    >>> model = {"North[3]":True, "P[3,4,1]":True, "P[3,3,1]":False, "West[1]":True, "GhostScary":True, "West[3]":False, "South[2]":True, "East[1]":False}
    >>> actions = ['North', 'South', 'East', 'West']
    >>> plan = extractActionSequence(model, actions)
    >>> print plan
    ['West', 'South', 'North']
    """
    "*** YOUR CODE HERE ***"
    action_model = {}
    for key, value in model.items():
        if value:
            token = logic.PropSymbolExpr.parseExpr(key)
            if token[0] in actions:
                action_model[int(token[1])] = token[0]
    action_model.items().sort()
    return action_model.values()

def pacmanSuccessorStateAxioms(x, y, t, walls_grid):
    """
    Successor state axiom for state (x,y,t) (from t-1), given the board (as a 
    grid representing the wall locations).
    Current <==> (previous position at time t-1) & (took action to move to x, y)
    """
    "*** YOUR CODE HERE ***"
    lst = []
    for i in [(-1, 0, "East"), (1, 0, "West"), (0, -1, "North"), (0, 1, "South")]:
        xt, yt = x + i[0], y + i[1]
        if not walls_grid[xt][yt]:
            lst.append(logic.PropSymbolExpr(pacman_str, xt, yt, t - 1) & logic.PropSymbolExpr(i[2], t - 1))
    if lst:
        return logic.PropSymbolExpr(pacman_str, x, y, t) % logic.disjoin(lst)
    else:
        return logic.PropSymbolExpr(pacman_str, x, y, t) % logic.PropSymbolExpr(pacman_str, x, y, t - 1)


def positionLogicPlan(problem):
    """
    Given an instance of a PositionPlanningProblem, return a list of actions that lead to the goal.
    Available actions are game.Directions.{NORTH,SOUTH,EAST,WEST}
    Note that STOP is not an available action.
    """
    walls = problem.walls
    width, height = problem.getWidth(), problem.getHeight()
    
    "*** YOUR CODE HERE ***"
    kb = []
    actions = ['North', 'South', 'East', 'West']
    start = problem.getStartState()
    goal = problem.getGoalState()
    
    for x in range(1, width + 1):
        for y in range(1, height + 1):
            if (x, y) == start:
                kb.append(logic.PropSymbolExpr(pacman_str, x, y, 0))
            else:
                kb.append(~logic.PropSymbolExpr(pacman_str, x, y, 0))

    for t in range(50):
        
        action = []
        for a in actions:
            action.append(logic.PropSymbolExpr(a, t)) 
        kb.append(exactlyOne(action))
        #print(kb)
        for x in range(1, width + 1):
            for y in range(1, height + 1):
                kb.append(pacmanSuccessorStateAxioms(x, y, t + 1, walls))

        kb.append(logic.PropSymbolExpr(pacman_str, goal[0], goal[1], t))

        model = findModel(logic.conjoin(kb))
        if model:
            return extractActionSequence(model, actions)

        kb.pop()


def foodLogicPlan(problem):
    """
    Given an instance of a FoodPlanningProblem, return a list of actions that help Pacman
    eat all of the food.
    Available actions are game.Directions.{NORTH,SOUTH,EAST,WEST}
    Note that STOP is not an available action.
    """
    walls = problem.walls
    width, height = problem.getWidth(), problem.getHeight()

    "*** YOUR CODE HERE ***"
    kb = []
    actions = ['North', 'South', 'East', 'West']
    start = problem.getStartState()[0]
    food = problem.getStartState()[1]
    
    for x in range(1, width + 1):
        for y in range(1, height + 1):
            if (x, y) == start:
                kb.append(logic.PropSymbolExpr(pacman_str, x, y, 0))
            else:
                kb.append(~logic.PropSymbolExpr(pacman_str, x, y, 0))

    for t in range(50):
        
        lst = []
        for a in actions:
            lst.append(logic.PropSymbolExpr(a, t)) 
        kb.append(exactlyOne(lst))
        #print(kb)
        for x in range(1, width + 1):
            for y in range(1, height + 1):
                kb.append(pacmanSuccessorStateAxioms(x, y, t + 1, walls))

        for x in range(1, width + 1):
            for y in range(1, height + 1):
                if food[x][y]:
                    lst = []
                    for t in range(t + 1):
                        lst.append(logic.PropSymbolExpr(pacman_str, x, y, t))
                    kb.append(atLeastOne(lst))

        model = findModel(logic.conjoin(kb))
        if model:
            return extractActionSequence(model, actions)

        for x in range(1, width + 1):
            for y in range(1, height + 1):
                if food[x][y]:
                    kb.pop()


def ghostPositionSuccessorStateAxioms(x, y, t, ghost_num, walls_grid):
    """
    Successor state axiom for patrolling ghost state (x,y,t) (from t-1).
    Current <==> (causes to stay) | (causes of current)
    GE is going east, ~GE is going west 
    """
    pos_str = ghost_pos_str+str(ghost_num)
    east_str = ghost_east_str+str(ghost_num)

    "*** YOUR CODE HERE ***"
    lst = []

    if not walls_grid[x - 1][y]:
        lst.append(logic.PropSymbolExpr(pos_str, x - 1, y, t - 1) & logic.PropSymbolExpr(east_str, t - 1))
    if not walls_grid[x + 1][y]:
        lst.append(logic.PropSymbolExpr(pos_str, x + 1, y, t - 1) & ~logic.PropSymbolExpr(east_str, t - 1))

    if lst:
        return logic.PropSymbolExpr(pos_str, x, y, t) % logic.disjoin(lst)
    else:
        return logic.PropSymbolExpr(pos_str, x, y, t) % logic.PropSymbolExpr(pos_str, x, y, t - 1)


def ghostDirectionSuccessorStateAxioms(t, ghost_num, blocked_west_positions, blocked_east_positions):
    """
    Successor state axiom for patrolling ghost direction state (t) (from t-1).
    west or east walls.
    Current <==> (causes to stay) | (causes of current)
    """
    pos_str = ghost_pos_str+str(ghost_num)
    east_str = ghost_east_str+str(ghost_num)

    "*** YOUR CODE HERE ***"
    wlst, elst = [], []

    for (x, y) in blocked_east_positions:
        elst.append(~logic.PropSymbolExpr(pos_str, x, y, t))
    for (x, y) in blocked_west_positions:
        wlst.append(logic.PropSymbolExpr(pos_str, x, y, t))

    if not elst and wlst:
        return logic.PropSymbolExpr(east_str, t) % (logic.PropSymbolExpr(east_str, t - 1) | (~logic.PropSymbolExpr(east_str, t - 1) & logic.disjoin(wlst)))
    elif elst and not wlst:
        return logic.PropSymbolExpr(east_str, t) % (logic.PropSymbolExpr(east_str, t - 1) & logic.conjoin(elst))
    elif not elst and not wlst:
        return logic.PropSymbolExpr(east_str, t) % logic.PropSymbolExpr(east_str, t - 1)
    else:
        return logic.PropSymbolExpr(east_str, t) % ((logic.PropSymbolExpr(east_str, t - 1) & logic.conjoin(elst)) | (~logic.PropSymbolExpr(east_str, t - 1) & logic.disjoin(wlst)))


def pacmanAliveSuccessorStateAxioms(x, y, t, num_ghosts):
    """
    Successor state axiom for patrolling ghost state (x,y,t) (from t-1).
    Current <==> (causes to stay) | (causes of current)
    """
    ghost_strs = [ghost_pos_str+str(ghost_num) for ghost_num in xrange(num_ghosts)]

    "*** YOUR CODE HERE ***"
    lst1, lst2 = [], []

    for g in ghost_strs:
        lst1.append(logic.PropSymbolExpr(g, x, y, t))
        lst2.append(logic.PropSymbolExpr(g, x, y, t - 1))

    if len(ghost_strs):
        causes = (logic.PropSymbolExpr(pacman_str, x, y, t) & logic.disjoin(lst2)) | logic.conjoin((logic.PropSymbolExpr(pacman_str, x, y, t), ~logic.disjoin(lst2), logic.disjoin(lst1)))
    else:
        return ~logic.PropSymbolExpr(pacman_alive_str, t) % ~logic.PropSymbolExpr(pacman_alive_str, t - 1)
    #print logic.PropSymbolExpr(pacman_alive_str, t) % logic.conjoin(logic.PropSymbolExpr(pacman_alive_str, t - 1), cause)
    return ~logic.PropSymbolExpr(pacman_alive_str, t) % ((logic.PropSymbolExpr(pacman_alive_str, t - 1) & causes) | ~logic.PropSymbolExpr(pacman_alive_str, t - 1)) 


def foodGhostLogicPlan(problem):
    """
    Given an instance of a FoodGhostPlanningProblem, return a list of actions that help Pacman
    eat all of the food and avoid patrolling ghosts.
    Ghosts only move east and west. They always start by moving East, unless they start next to
    and eastern wall. 
    Available actions are game.Directions.{NORTH,SOUTH,EAST,WEST}
    Note that STOP is not an available action.
    """
    walls = problem.walls
    width, height = problem.getWidth(), problem.getHeight()

    "*** YOUR CODE HERE ***"
    kb = []
    actions = ['North', 'South', 'East', 'West']
    start = problem.getStartState()[0]
    food = problem.getStartState()[1]
    ghosts = problem.getGhostStartStates()
    num_ghosts = len(ghosts)
    blocked_west_positions, blocked_east_positions = [], []
    ghosts_pos = []

    # init KB
    for i in range(len(ghosts)):
        ghosts_pos.append(ghosts[i].getPosition())    

    for x in range(1, width + 1):
        for y in range(1, height + 1):
            if (x, y) == start:
                kb.append(logic.PropSymbolExpr(pacman_str, x, y, 0))
            else:
                kb.append(~logic.PropSymbolExpr(pacman_str, x, y, 0))
            for i in range(len(ghosts)):
                pos_str = ghost_pos_str + str(i)
                if (x, y) == ghosts_pos[i]:
                    kb.append(logic.PropSymbolExpr(pos_str, x, y, 0))
                else:
                    kb.append(~logic.PropSymbolExpr(pos_str, x, y, 0))
            if walls[x + 1][y]:
                blocked_east_positions.append((x, y))
            if walls[x - 1][y]:                    
                blocked_west_positions.append((x, y))

    for i in range(len(ghosts)):
        east_str = ghost_east_str + str(i)
        if ghosts_pos[i] not in blocked_east_positions:
            kb.append(logic.PropSymbolExpr(east_str, 0))
        else:
            kb.append(~logic.PropSymbolExpr(east_str, 0))

    kb.append(logic.PropSymbolExpr(pacman_alive_str, 0))
    #print(kb)

    # loop each time
    for t in range(50):
        #print(t)

        #exactly one action each time
        lst = []
        for a in actions:
            lst.append(logic.PropSymbolExpr(a, t)) 
        kb.append(exactlyOne(lst))

        # SSAs
        for x in range(1, width + 1):
            for y in range(1, height + 1):
                kb.append(logic.to_cnf(pacmanSuccessorStateAxioms(x, y, t + 1, walls)))
                kb.append(logic.to_cnf(pacmanAliveSuccessorStateAxioms(x, y, t + 1, num_ghosts)))
                for i in range(len(ghosts)):
                    kb.append(logic.to_cnf(ghostPositionSuccessorStateAxioms(x, y, t + 1, i, walls)))
        
        for i in range(len(ghosts)):
            kb.append(logic.to_cnf(ghostDirectionSuccessorStateAxioms(t + 1, i, blocked_west_positions, blocked_east_positions)))

        # goal KB
        for x in range(1, width + 1):
            for y in range(1, height + 1):
                if food[x][y]:
                    lst = []
                    for t in range(t + 1):
                        lst.append(logic.PropSymbolExpr(pacman_str, x, y, t))
                    kb.append(atLeastOne(lst))

        # whether satisfy the model
        model = logic.pycoSAT(logic.conjoin(kb))
        if model:
            return extractActionSequence(model, actions)

        # pop goal KB
        for x in range(1, width + 1):
            for y in range(1, height + 1):
                if food[x][y]:
                    kb.pop()


# Abbreviations
plp = positionLogicPlan
flp = foodLogicPlan
fglp = foodGhostLogicPlan

# Some for the logic module uses pretty deep recursion on long expressions
sys.setrecursionlimit(100000)
    