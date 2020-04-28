import copy
import itertools
import random
import statistics

import matplotlib.pyplot as plt
import numpy


class State:
    def __init__(self, matrixSize: int, empty: bool = False):
        self.matrixSize = matrixSize
        self.matrix = []
        if not empty:
            available = random.sample([x for x in itertools.product(
                range(1, self.matrixSize + 1), range(1, self.matrixSize + 1))],
                                      self.matrixSize ** 2)
            for _ in range(self.matrixSize):
                to_add, available = available[:self.matrixSize], available[self.matrixSize:]
                self.matrix.append(to_add)
        else:
            self.matrix = [[] for _ in range(self.matrixSize)]

    def fitness(self):
        wrongCases = 0
        # For lines
        for x in range(0, 2):
            for i in range(self.matrixSize):
                for j in range(self.matrixSize - 1):
                    for k in range(j + 1, self.matrixSize):
                        if self.matrix[i][j][x] == self.matrix[i][k][x]:
                            wrongCases += 1
        # For columns
        for x in range(0, 2):
            for j in range(self.matrixSize):
                for i in range(self.matrixSize - 1):
                    for k in range(i + 1, self.matrixSize):
                        if self.matrix[i][j][x] == self.matrix[k][j][x]:
                            wrongCases += 1
        return wrongCases * -1

    def isSolution(self):
        return self.fitness() == 0

    def __str__(self) -> str:
        res = ''
        for i in range(self.matrixSize):
            for j in range(self.matrixSize):
                res += str(self.matrix[i][j]) + '\t'
            res += '\n'
        return res


class Problem:
    def __init__(self, matrixSize, populationSize, noGenerations, mutation: float, crossover: float):
        self.matrixSize = matrixSize
        self.populationSize = populationSize
        self.noGenerations = noGenerations
        self.mutation = mutation
        self.crossover = crossover
        self.currentGen = 0

        self.population = [State(self.matrixSize) for _ in range(self.populationSize)]
        self.currentState = State(self.matrixSize)

    def maxFitness(self):
        return max(x.fitness() for x in self.population)

    def avgFitness(self):
        return statistics.mean(x.fitness() for x in self.population)

    def isSolved(self):
        return self.maxFitness() == 0

    def crtFitness(self):
        return self.currentState.fitness()

    def bestSol(self):
        maxim = self.maxFitness()
        for x in self.population:
            if x.fitness() == maxim:
                return x

    def generateNeighbour(self):
        i1, i2, j1, j2 = [random.randint(0, self.matrixSize - 1) for _ in range(4)]
        newState = copy.deepcopy(self.currentState)
        newState.matrix[i1][j1], newState.matrix[i2][j2] = newState.matrix[i2][j2], newState.matrix[i1][j1]
        return newState

    def generateNeighbours(self):
        return [self.generateNeighbour() for _ in range(self.populationSize)]

    def hillClimbingNextStep(self):
        if self.currentGen % 200 == 0:
            self.currentState = State(self.matrixSize)
        else:
            nextGen = self.generateNeighbours()
            maxFitness = max(x.fitness() for x in nextGen)
            if maxFitness > self.currentState.fitness():
                for x in nextGen:
                    if x.fitness() == maxFitness:
                        self.currentState = x
                        break
        self.currentGen += 1

    def mutateState(self, state):
        if random.uniform(0, 1) > self.mutation:
            return state
        else:
            lineToReplace = random.randint(0, self.matrixSize - 1)
            newLine = list(zip(numpy.random.permutation(list(range(1, self.matrixSize + 1))),
                               numpy.random.permutation(list(range(1, self.matrixSize + 1)))))
        state.matrix[lineToReplace] = newLine
        return state

    def breedPair(self, state1, state2):
        if random.uniform(0, 1) > self.crossover:
            if random.uniform(0, 1) > 0.5:
                return state1
            else:
                return state2
        result = State(self.matrixSize, True)
        for newIndex in range(self.matrixSize):
            index = random.randint(0, self.matrixSize - 1)
            if random.uniform(0, 1) > 0.5:
                result.matrix[newIndex] = state1.matrix[index]
            else:
                result.matrix[newIndex] = state2.matrix[index]
        return result

    def makeNewGeneration(self):
        populationWeights = [state.fitness() for state in self.population]
        offset = min(populationWeights)
        populationWeights = [z - offset for z in populationWeights]

        newPopulation = [
            random.choices(self.population, weights=populationWeights, k=self.populationSize * 2)
            for _ in range(2)]
        newPopulation = [
            self.mutateState(self.breedPair(newPopulation[0][index], newPopulation[1][index]))
            for
            index in range(len(newPopulation[0]))]

        offset = min(x.fitness() for x in newPopulation)
        newPopulation.sort(key=lambda x: x.fitness() - offset, reverse=True)
        self.population = newPopulation[:self.populationSize]
        self.currentGen += 1


from collections import namedtuple

CurrentState = namedtuple('CurrentState', 'finished currentGen maxFitness avgFitness bestSolution')


def _state_to_str(state: CurrentState) -> str:
    res = ''
    if state.finished:
        res += 'Finished!\n'
    else:
        res += 'Not finished!\n'
    res += f'Current Gen = {state.currentGen}\n'
    res += f'Max Fitness = {state.maxFitness}\n'
    res += f'Avg Fitness = {state.avgFitness}\n'
    res += f'Best Solution : \n{state.bestSolution}\n'
    return res


CurrentState.__str__ = _state_to_str
CurrentState.__repr__ = CurrentState.__str__


class Controller:
    def __init__(self, matrixSize, populationSize, noGenerations, mutation=0.0, crossover=0.0):
        self.matrixSize = matrixSize
        self.populationSize = populationSize
        self.noGenerations = noGenerations
        self.mutation = mutation
        self.crossover = crossover
        self.problem = Problem(self.matrixSize, self.populationSize, self.noGenerations, self.mutation, self.crossover)

    def hillClimbingAlgorithm(self):
        self.problem.hillClimbingNextStep()
        if self.problem.currentGen == self.noGenerations or self.problem.currentState.fitness() == 0:
            return CurrentState(True,
                                self.problem.currentGen,
                                self.problem.maxFitness(),
                                self.problem.avgFitness(),
                                self.problem.bestSol())
        else:
            return CurrentState(False,
                                self.problem.currentGen,
                                self.problem.maxFitness(),
                                self.problem.avgFitness(),
                                self.problem.bestSol())

    def currentState(self):
        return CurrentState(self.problem.isSolved(), self.problem.currentGen,
                            self.problem.maxFitness(), self.problem.avgFitness(),
                            self.problem.bestSol())

    def evolutionaryAlgorithm(self):
        self.problem.makeNewGeneration()
        if self.problem.currentGen == self.noGenerations or self.problem.isSolved():
            return CurrentState(True, self.problem.currentGen, self.problem.maxFitness(),
                                self.problem.avgFitness(), self.problem.bestSol())
        else:
            return CurrentState(False, self.problem.currentGen,  self.problem.maxFitness(),
                                self.problem.avgFitness(), self.problem.bestSol())


class Console:
    def __init__(self):
        self.algorithm = int(input('Enter 1 for Hill Climbing or 2 for Evolutionary\n'))
        self.matrixSize = int(input('Matrix size = '))
        self.populationSize = int(input('Population size = '))
        self.maxGenerations = int(input('Maximum generations = '))
        if self.algorithm == 1:
            self.controller = Controller(self.matrixSize, self.populationSize, self.maxGenerations)
            self.run(1)
        elif self.algorithm == 2:
            self.mutationChance = float(input('Mutation chance [0, 1] = '))
            self.crossoverChance = float(input('Crossover chance [0, 1] = '))
            self.controller = Controller(self.matrixSize, self.populationSize, self.maxGenerations, self.mutationChance,
                                         self.crossoverChance)
            self.run(2)
        else:
            print("Please enter just 1 or 2.")
            return

    def drawPlot(self, max, avg):
        plt.xlabel("Trials")
        plt.ylabel("Fitness")
        plt.plot(max, label='Maximum Fitness')
        plt.plot(avg, label='Average Fitness')
        plt.legend(loc='upper left', frameon=False)
        plt.show()

    def run(self, algorithm):
        # choose = int(input("Enter 1 for just 1 step, 2 for all steps or 0 for Stop"))
        maxFitnessList = []
        avgFitnessList = []

        while True:
            if algorithm == 1:
                result = self.controller.hillClimbingAlgorithm()
            else:
                result = self.controller.evolutionaryAlgorithm()

            maxFitnessList.append(result.maxFitness)
            avgFitnessList.append(result.avgFitness)


            print(result)
            if result.finished:
                self.drawPlot(maxFitnessList, avgFitnessList)
                break


console = Console()
