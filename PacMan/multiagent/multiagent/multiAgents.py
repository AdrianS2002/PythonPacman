# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        "*** YOUR CODE HERE ***"
        "distanta de la pacman la mancare"
        minFoodDistance = -1
        newFoodList = newFood.asList()
        for food in newFoodList:
            distance = util.manhattanDistance(newPos, food)
            if minFoodDistance >= distance or minFoodDistance == -1:
                minFoodDistance = distance
        """distanta de la pacman la fantome"""
        distancesGhosts = 1
        proximityGhosts = 0
        for ghost_state in successorGameState.getGhostPositions():
            distance = util.manhattanDistance(newPos, ghost_state)
            distancesGhosts += distance
            if distance <= 1:
                proximityGhosts += 1
        return successorGameState.getScore() + (1 / float(minFoodDistance)) - (1 / float(distancesGhosts)) - proximityGhosts

        return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***" 
        def minFunc(gameState,agentID,depth):
            actions=gameState.getLegalActions(agentID)
            if len(actions) == 0:
                return(self.evaluationFunction(gameState),None)
            minVal=999999                                                                                 ###As we see in contrast with max we begin from +infinte
            actiune=None
            for act in actions:
                if(agentID==gameState.getNumAgents() -1): 
                    sucsValue=maxFunc(gameState.generateSuccessor(agentID,act),depth + 1)
                else:
                    sucsValue=minFunc(gameState.generateSuccessor(agentID,act),agentID+1,depth)        ###We are doing exactly the opposite from the max "function"
                sucsValue=sucsValue[0]
                if(sucsValue<minVal):
                    minVal,actiune=sucsValue,act
            return(minVal,actiune)
        
        def maxFunc(gameState,depth):
            actions=gameState.getLegalActions(0)
            if len(actions)==0 or gameState.isWin() or gameState.isLose() or depth==self.depth:             ###The trvial situations(state)
                return(self.evaluationFunction(gameState),None)
            maxVal=-999999999                                                                              ###We are trying to implement the 2 sides of the minimax algorithm the max and the min
            actiune=None
            for act in actions:                                                                          ###In that way that the 2 functions are calling each other is like building the tree(diagrams from tha class)
                sucsValue=minFunc(gameState.generateSuccessor(0,act),1,depth)                          #We have the available moves and we are seeking for the "best" one
                sucsValue=sucsValue[0]                                                                      #It is working exactly as the theory of minimax algorithm commands
                if(sucsValue>maxVal):                                                                            #Here we have as start -infinite
                    maxVal,actiune=sucsValue,act
            return(maxVal,actiune)

        
        maxFunc=maxFunc(gameState,0)[1]
        return maxFunc                                                                                    ###We are starting from the max and it goes as a tree max min max min


        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def PacmanOrAgent(self, gameState, alpha, beta, depth, agent):
        if agent >= gameState.getNumAgents():
            depth    = depth + 1
            agent = 0
        if (depth == self.depth or gameState.isWin() or gameState.isLose()):
            return self.evaluationFunction(gameState)
        if agent == 0:
            return self.maxFunc(gameState, alpha, beta , depth, agent) #pacman
        else: 
            return self.minFunc(gameState, alpha, beta,depth, agent )  #fantoma

    def maxFunc(self, gameState, alpha, beta, depth, agent):
        maxVal = -999999
        legalActions = gameState.getLegalActions(agent)
        for action in legalActions:
            successorGameState = gameState.generateSuccessor(agent, action)
            maxVal = max(maxVal, self.PacmanOrAgent(successorGameState, alpha, beta, depth, agent + 1))
            if maxVal > beta:
              return maxVal
            alpha = max(alpha, maxVal)
        return maxVal
    
    def minFunc(self, gameState, alpha, beta, depth, agent):
        minVal = 999999
        legalActions = gameState.getLegalActions(agent)
        for action in legalActions:
            successorGameState = gameState.generateSuccessor(agent, action)
            minVal = min(minVal,self.PacmanOrAgent(successorGameState, alpha, beta, depth, agent + 1))
            if minVal < alpha:
              return minVal
            beta = min(beta, minVal)
        return minVal

    def getAction(self, gameState):
        alpha = -999999
        beta = 999999
        agent = 0
        legalActions = gameState.getLegalActions(agent)
        for action in legalActions:
            successorGameState = gameState.generateSuccessor(agent, action)
            val = self.PacmanOrAgent(successorGameState, alpha, beta, 0, 1)
            if val > alpha:
              alpha = val
              best_action = action
        return best_action

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        def expectMax(gameState, agent, depth=0):
            chIndex = gameState.getNumAgents() - 1
            if agent == self.index:
                value = -9999999999999999999
            else:
                value = 0
            if (gameState.isLose() or gameState.isWin() or depth == self.depth):
                return [self.evaluationFunction(gameState)]
            elif agent == chIndex:
                depth += 1
                childIndex = self.index
            else:
                childIndex = agent + 1

            legalActionList = gameState.getLegalActions(agent)
            
            bestAction = None    
            numAction = len(legalActionList) 

            for legalAction in legalActionList:
                successorGameState = gameState.generateSuccessor(agent, legalAction)
                expectedMax = expectMax(successorGameState, childIndex, depth)[0]
                if agent == self.index:
                    if expectedMax > value:
                    
                        value = expectedMax
                        bestAction = legalAction
                else:
                    #value = value + prob(move) * nxt_val
                    value = value + ((1.0/numAction) * expectedMax)
            return value, bestAction

        bestScoreActionPair = expectMax(gameState, self.index)
        bestMove =  bestScoreActionPair[1]
        return bestMove
        
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
  """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: Our evaluation function uses distance between ghosts and pacman as well as distance between pacman and capsules
                 along with current game score. The distance between ghost and pacman is subtracted from current score and distance
                 between capsules and pacman is added to current score. 
  """
  "*** YOUR CODE HERE ***"
    
  food = currentGameState.getCapsules()   
  ghostStates = currentGameState.getGhostStates()
  pacman_pos = currentGameState.getPacmanPosition()
  current_score = currentGameState.getScore()
  foodScore = 0

  #Calculate the distance between pacman and capsules in game using Manhattan distance.
  #Check if capsule list is not empty.
  if(len(food) != 0):
    #Use manhattan distance formula
    for point in food:
      if  min([manhattanDistance(point, pacman_pos)]) == 0 :
        foodScore = float(1)/ min([manhattanDistance(point, pacman_pos)])
      else:
        foodScore = -1
        
  #Calculate distance between ghosts and pacman using Manhattan distance.s        
  for ghost in ghostStates:
    ghostX= (ghost.getPosition()[0])
    ghostY = (ghost.getPosition()[1])
    ghostPos = tuple([ghostX, ghostY])
  #Evaluation function returms following scores.
  return current_score  - (1.0/1+manhattanDistance(pacman_pos, ghostPos))  + (1.0/1+foodScore)


# Abbreviation
better = betterEvaluationFunction 
