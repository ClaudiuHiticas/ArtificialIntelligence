from collections import defaultdict
from itertools import permutations, product
from random import choice, random

class Graph:
    def __init__(self):
        self.graph = defaultdict(dict)

    def __getitem__(self, item):
        return self.graph[item[0]][item[1]]

    def __setitem__(self, key, value):
        self.graph[key[0]][key[1]] = value

    def addEdge(self, x, y):
        self.graph[x][y] = 1
        self.graph[y][x] = 1

    def getNodes(self):
        return [index for index in self.graph.keys()]

    def getNeighbours(self, x):
        return [index for index in self.graph[x].keys()]



class Ant:
    def __init__(self, size, S, T, graph, alpha, beta, trace, q0):
        self.size = size
        self.S = S
        self.T = T
        self.graph = graph
        self.alpha = alpha
        self.beta = beta
        self.trace = trace
        self.q0 = q0
        self.path = [choice(self.graph.getNodes())]

    def getPath(self):
        return self.path

    def nextMoves(self, thisPath):
        if len(thisPath) == self.size:
            return []

        last = thisPath[-1]
        pos = []

        for neigh in self.graph.getNeighbours(last):
            path = thisPath[:]
            path.append(neigh)
            pos.append(path)

        valid = []
        for p in pos:
            if self.repeat(p) == 0:
                valid.append(p)

        return valid

    def repeat(self, path):
        cnt = 0
        for i in range(self.size):
            visited = []
            for j in range(len(path)):
                if path[j][i] not in visited:
                    visited.append(path[j][i])
                else:
                    cnt += 1
        return cnt

    def fit(self, trace, path):
        return trace ** self.alpha * len(self.nextMoves(path)) ** self.beta


    def finished(self):
        return len(self.path) == self.size

    def addMove(self):
        if self.finished():
            return

        paths = self.nextMoves(self.path)
        paths = sorted(paths, key=lambda x: self.fit(self.graph[self.path[-1], x[-1]], x), reverse=True)

        if random() < self.q0:
            path = paths[0]
        else:
            path = choice(paths)

        self.graph[self.path[-1], path[-1]] += self.trace
        self.path = path


def generateGraph(n):
    vals = [i for i in range(1, n + 1)]
    perms = permutations(vals)
    g = Graph()
    for x, y in product(perms, repeat=2):
        g.addEdge(x, y)
    return g

class Problem:
    def __init__(self, size, iter, nrAnts, trace, alpha, beta, q0):
        self._size = size
        self._S = [i for i in range(1, size + 1)]
        self._T = [i for i in range(1, size + 1)]
        self._iter = iter
        self._nrAnts = nrAnts
        self._trace = trace
        self._alpha = alpha
        self._beta = beta
        self._q0 = q0

    @property
    def size(self):
        return self._size
    @property
    def iter(self):
        return self._iter
    @property
    def nrAnts(self):
        return self._nrAnts
    @property
    def trace(self):
        return self._trace
    @property
    def alpha(self):
        return self._alpha
    @property
    def beta(self):
        return self._beta
    @property
    def q0(self):
        return self._q0

def toSquares(solution):
    list = []
    h = len(solution) // 2
    for i in range(h):
        for j in range(h):
            matrix = [solution[i][j], solution[i + h][j]]
            list.append(matrix)
    return list


def fitness(solution):
    cnt = 0
    h = len(solution) // 2
    for i in range(len(solution)):
        rows = []
        columns = []
        for j in range(h):
            if solution[i][j] not in rows:
                rows.append(solution[i][j])
            else:
                cnt += 1
            if i >= h:
                if solution[j + h][i // 2] not in columns:
                    columns.append(solution[j + h][i // 2])
                else:
                    cnt += 1
            else:
                if solution[j][i // 2] not in columns:
                    columns.append(solution[j][i // 2])
                else:
                    cnt += 1

    geno = toSquares(solution)
    n = len(geno)
    for i in range(n):
        if geno[i] in geno[i + 1:]:
            cnt += 1

        return cnt


class Controller:
    def __init__(self, problem):
        self.prb = problem

    def execute(self):
        idx = 0

        firstGraph = generateGraph(self.prb.size)
        secondGraph = generateGraph(self.prb.size)

        S = [idx for idx in range(self.prb.size)]
        T = [idx for idx in range(self.prb.size)]

        prb = self.prb
        firstAnt = Ant(prb.size, S, T, firstGraph, prb.alpha, prb.beta, prb.trace, prb.q0)
        secondAnt = Ant(prb.size, S, T, secondGraph, prb.alpha, prb.beta, prb.trace, prb.q0)

        firstPopulation = [firstAnt for _ in range(prb.nrAnts)]
        secondPopulation = [secondAnt for _ in range(prb.nrAnts)]


        while idx < prb.iter and (not firstPopulation[0].finished() or not secondPopulation[0].finished()):
            for x in firstPopulation:
                x.addMove()
            for x in secondPopulation:
                x.addMove()
            idx += 1

        firstSol = [a.getPath() for a in firstPopulation]
        secondSol = [a.getPath() for a in secondPopulation]

        solutions = []

        for s1, s2 in product(firstSol, secondSol):
            solution = s1 + s2
            solutions.append(solution)

        solutions = sorted(solutions, key=lambda x: fitness(x))

        bestSol = solutions[0]
        lenBestSol = len(bestSol) // 2
        matrix = ""
        for i in range(lenBestSol):
            for j in range(lenBestSol):
                matrix = matrix + "(" + str(bestSol[i][j]) + "," + str(bestSol[i + lenBestSol][j]) + ") "
            matrix += '\n'

        print("The solution is:\n" + matrix + "\nBest fitness: " + str(fitness(bestSol)))




def main():
    matrixSize = int(input('Matrix size = '))
    iter = int(input('Iterations = '))
    nrAnts = int(input('Number of Ants = '))
    trace = int(input('Trace = '))
    alpha = float(input('Alpha = '))
    beta = float(input('Beta = '))
    q0 = float(input('Q0 = '))

    # prb = Problem(4, 20, 1, 1, 1.9, 0.9, 0.5)
    prb = Problem(matrixSize, iter, nrAnts, trace, alpha, beta, q0)
    ctrl = Controller(prb)
    ctrl.execute()

main()