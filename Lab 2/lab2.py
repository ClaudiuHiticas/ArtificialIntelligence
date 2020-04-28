import numpy
import copy
import time

class State:
    def __init__(self, size):
        self.size = size
        self.matrix = numpy.zeros((size, size), dtype=int).tolist()

    def getMatrix(self):
        return self.matrix

    def setMatrix(self, m):
        self.matrix = m

    def getDim(self):
        return self.size

    def setElem(self, i, j):
        self.matrix[i][j] = 1

    def isPositionValid(self, line, col):
        for i in range(self.size):
            for j in range(self.size):
                if self.matrix[i][j] == 1:
                    if i == line or j == col or abs(line - i) == abs(col - j):
                        return False
        return True

    def isMatrixValid(self):
        for i in range(self.size):
            for j in range(self.size):
                if self.matrix[i][j] == 1 and not self.isPositionValid(i, j):
                    return False
        return True
    
    def nextConf(self):
        result = []
        for i in range(self.size):
            for j in range(self.size):
                if self.matrix[i][j] != 1 and self.isPositionValid(i, j):
                    newMatrix = copy.deepcopy(self.matrix)
                    newMatrix[i][j] = 1
                    newState = State(self.size)
                    newState.matrix = newMatrix
                    result.append(newState)
        return result

    def __eq__(self, value):
        if not isinstance(value, State):
            return False
        if self.size != value.size:
            return False
        for i in range(self.size):
            for j in range(self.size):
                if self.matrix[i][j] != value.matrix[i][j]:
                    return False
        return True

    def __str__(self):
        s = ""
        for i in self.matrix:
            for j in i:
                s += str(j)
                s += " "
            s += "\n"
        return s



class Problem:
    def __init__(self, initialState):
        self.initialState = initialState
    
    def expend(state):
        if(any(0 in line for line in state.getMatrix())):
            return state.nextConf()
        else:
            return None

    def heuristic(state):
        if not state.isMatrixValid():
            return -1
        score = 0
        for i in range(state.getDim()):
            for j in range(state.getDim()):
                if state.matrix[i][j] == 1:
                    score += 1
        return score

    def isSolved(state):
        suma = 0
        for i in state.getMatrix():
            suma += sum(i)
        if suma == state.getDim():
            return True
        return False


class Controller:
    def __init__(self, problem):
        self.problem = problem
    
    def orderStates(self, states):
        return sorted(states, key=lambda state:Problem.heuristic(state), reverse = True)

    def dfs(self):
        stack = [copy.deepcopy(self.problem.initialState)]

        while(stack):
            currentState = stack.pop(0)
            if Problem.isSolved(currentState):
                return currentState
            
            expendState = Problem.expend(currentState)
            if expendState is not None:
                stack = expendState + stack
        return None

    def gbfs(self):
        queue = [copy.deepcopy(self.problem.initialState)]

        while(queue):
            currentState = queue.pop(0)
            if Problem.isSolved(currentState):
                return currentState
            
            expendState = Problem.expend(currentState)
            if expendState is not None:
                queue = expendState + queue
            queue = Controller.orderStates(self, queue)
        return None


class UI:
    def __init__(self):
        self.size = 4
        self.initialState = State(self.size)
        self.problem = Problem(self.initialState)
        self.controller = Controller(self.problem)

    def showMenu(self):
        print('''Please choose your option:
    1. Change the size (4 as default)
    2. DFS
    3. Greedy
    0. Exit
        ''')
    
    def solve(self, alg):
        start = time.time()
        result = None
        if(alg == "dfs"):
            result = self.controller.dfs()
        elif(alg == "gbfs"):
            result = self.controller.gbfs()
        stop = time.time()
        if result != None:
            print(result)
            print("Time: ", stop - start)
        else:
            print("Solution not found")

    def mainMenu(self):
        while True:
            self.showMenu()
            n = int(input("n = "))
            if n == 1:
                newSize = None
                try:
                    newSize = int(input("New size:"))
                except(ValueError, TypeError):
                    print('Invalid input.')
                self.size = newSize
                self.initialState = State(self.size)
                self.problem = Problem(self.initialState)
                self.controller = Controller(self.problem)
            elif n == 2:
                self.solve("dfs")
            elif n == 3:
                self.solve("gbfs")
            else:
                print("Bye bye!")
                break



def tests():
    states = [State(3), State(4)]
    problems = [Problem(st) for st in states]
    controllers = [Controller(pr) for pr in problems]

    assert (states[0].size == 3)
    assert (states[0].matrix == [[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    assert (states[0].isMatrixValid())
    for i in range(3):
        for j in range(3):
            assert (states[0].isPositionValid(i, j))
    assert (len(states[0].nextConf()) == 9)

    assert (Problem.isSolved(states[0]) == False)

    assert (controllers[0].dfs() is None)
    assert (controllers[0].gbfs() is None)

    assert (controllers[1].dfs() is not None)
    assert (controllers[1].gbfs() is not None)

    print("All tests prints with succes!")


tests()
ui = UI()
ui.mainMenu()