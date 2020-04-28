import itertools

from random import randint, sample, uniform, random
from statistics import mean, stdev


class Particle:
    def __init__(self, size):
        self._size = size
        self._position = self.createPos()
        self.evaluate()
        self._velocity = [[ 0 for _ in range(self._size)] for _ in range(self._size * 2)]
        self._bestPosition = self._position
        self._bestFitness = self._fitness

    def createPos(self):
        position = []
        available = sample([x for x in itertools.product(
            range(1, self._size + 1), range(1, self._size + 1))],
                                    self._size ** 2)
        for _ in range(self._size):
            to_add, available = available[:self._size], available[self._size:]
            position.append(to_add)
        return position


    def fit(self):
        wrongCases = 0
        # For lines
        for x in range(0, 2):
            for i in range(self._size):
                for j in range(self._size - 1):
                    for k in range(j + 1, self._size):
                        if self._position[i][j][x] == self._position[i][k][x]:
                            wrongCases += 1
        # For columns
        for x in range(0, 2):
            for j in range(self._size):
                for i in range(self._size - 1):
                    for k in range(i + 1, self._size):
                        if self._position[i][j][x] == self._position[k][j][x]:
                            wrongCases += 1
        return wrongCases


    def evaluate(self):
        self._fitness = self.fit()

    @property
    def size(self):
        return self._size

    @property
    def position(self):
        return self._position

    def getPosition(self):
        return self._position

    @property
    def fitness(self):
        return self._fitness

    @property
    def velocity(self):
        return self._velocity

    @velocity.setter
    def velocity(self, newV):
        self._velocity = newV

    @property
    def bestPosition(self):
        return self._bestPosition

    @property
    def bestFitness(self):
        return self._bestFitness

    @position.setter
    def position(self, newPosition):
        # print(newPosition)
        self._position = newPosition.copy()
        # automatic evaluation of particle's fitness
        self.evaluate()
        # automatic update of particle's memory
        if (self._fitness < self.bestFitness):
            self._bestPozition = self._position
            self._bestFitness  = self._fitness

    def isSolution(self):
        return self.fitness == 0
    
    def __lt__(self, other):
        return self.fitness < other.fitness
    
    def __gt__(self, other):
        return self.fitness > other.fitness

    def __eq__(self, other):
        return self.fitness == other.fitness

    def __str__(self):
        res = ''
        for i in range(self._size):
            for j in range(self._size):
                res += str(self._position[i][j]) + '\t'
            res += '\n'
        res += "Fitness: " + str(self.fitness)
        res += '\n'    
        return res

# def distance(first, second, k):
#     sum = 0
#     for j in range(len(first)):
#         sum += abs(first[j][k] - second[j][k])
#     return sum


class Population:
    def __init__(self, sizePop, sizeMat, sizeNeigh):
        self._sizePop = sizePop
        self._sizeMat = sizeMat
        self._sizeNeigh = sizeNeigh
        self._pop = [Particle(sizeMat) for _ in range(sizePop)]
        self._neighbors = self.selectNeighbors()

    def selectNeighbors(self):
        if self._sizeNeigh > self._sizePop:
            self._sizeNeigh = self._sizePop

        neighbors=[]
        for i in range(self._sizePop):
            localNeighbor=[]
            for j in range(self._sizeNeigh):
                x = randint(0, self._sizePop - 1)
                while x in localNeighbor:
                    x = randint(0, self._sizePop - 1)
                localNeighbor.append(x)
            neighbors.append(localNeighbor.copy())
        return neighbors

    def getPop(self):
        return self._pop

    def setPop(self, pop):
        self._pop = pop

    def getNeighbors(self):
        return self._neighbors

    def bestIndivid(self):
        return sorted(self._pop)[0]
    
    def __str__(self):
        res = ''
        for el in self._pop:
            res += str(el) + '\n'
        return res

    __repr__ = __str__


class Algorithm:
    def __init__(self, sizePop, sizeMat, sizeNeigh, c1, c2, w, noIter):
        self._sizePop = sizePop
        self._sizeMat = sizeMat
        self._sizeNeigh = sizeNeigh
        self._crtIter = 0
        self._noIter = noIter
        self._c1 = c1
        self._c2 = c2
        self._w = w
        self._pop = Population(self._sizePop, self._sizeMat, self._sizeNeigh)

    def bestNeighbours(self):
        neighs = self._pop.getNeighbors()
        bestNeighs = []
        for i in range(self._sizePop):
            bestNeighs.append(neighs[i][0])
            for j in range(1, len(neighs[i])):
                pop = self._pop.getPop()
                if pop[bestNeighs[i]] > pop[neighs[i][j]]:
                    bestNeighs[i] = neighs[i][j]
        return bestNeighs

    def newVelocity(self, bestNeighs):
        pop = self._pop.getPop()
        # print(pop[0])

        for i in range(self._sizePop):
            for j in range(self._sizeMat):
                for x in range(len(pop[0].velocity[j])):
                    newV = self._w * pop[i].velocity[j][x]
                    # newV = newV + self._c1 * uniform(0, 1) * distance(pop[bestNeighs[j]], pop[i]._position[j], x)
                    # newV = newV + self._c2 * uniform(0, 1) * distance(pop[i]._bestPosition[j], pop[i]._position[j], x)
                    newV = newV + self._c1 * random() * (pop[bestNeighs[i]].position[j][x] - pop[i].position[j][x])
                    newV = newV + self._c2 * random() * (pop[i].bestPosition[j][x] - pop[i].position[j][x])
                    pop[i].velocity[j][x] = newV
        return pop



    def updatePosition(self, pop):
        for i in range(self._sizePop):
            newPos = []
            for j in range(len(pop[0].velocity)):
                part = []
                for x in range(len(pop[0].velocity[j])):
                    var = pop[i].position[j][x] + pop[i].velocity[j][x]
                    if var > self._sizeMat:
                        var = self._sizeMat
                    else:
                        if var < 1:
                            var = 1
                    part.append(int(var))
                newPos.append(part)
            pop[i].position = newPos
        return pop

    def iteration(self):
        bestNeighs = self.bestNeighbours()
        print(bestNeighs)
        pop = self.newVelocity(bestNeighs)
        # pop = self.updatePosition(pop)

        # self._pop.setPop(pop)

    def run(self):
        while self._crtIter in range(self._noIter) and\
         self._pop.bestIndivid().fitness != 0:
            self._w /= self._crtIter + 1
            self.iteration()
            self._crtIter += 1
        return self._pop.bestIndivid()


def test():
    runs = 30
    iter = 1000
    pop_s = 40
    pso = Algorithm(pop_s, 4, 10, 1.5, 2.5, 1.0, iter)
    best = []
    for _ in range(runs):
        best.append(pso.run().fitness)
        pso = Algorithm(pop_s, 3, 10, 1.5, 2.5, 1.0, iter)
    avgPSO = mean(best)
    stdPSO = stdev(best)
    print(pso)
    print('Average: ' + avgPSO)
    print('Standard: ' + stdPSO)


# test()

runs = 30
iter = 1000
pop_s = 1
pso = Algorithm(pop_s, 4, 10, 1.5, 2.5, 1.0, iter)
print(pso.iteration())