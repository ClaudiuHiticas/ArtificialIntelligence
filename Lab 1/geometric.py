import numpy as np
import copy
import random
import math


def readFromFile(file):
    lines, columns, noShapes, shapes = None, None, None, []
    with open(file, 'r') as in_file:
        lines = int(in_file.readline().rstrip())
        columns = int(in_file.readline().rstrip())
        noShapes = int(in_file.readline().rstrip())
        for _ in range(noShapes):
            shape = []
            for _ in range(lines):
                read_line = in_file.readline().rstrip()
                line = list(read_line.split())
                line = [int(elem) for elem in line]
                shape.append(line)
            shapes.append(shape)
            _ = in_file.readline()
    # print(shapes)
    return lines, columns, noShapes, shapes
    

def printMatrix(lines, columns, shape):
    for i in range(lines):
        for j in range(columns):
            print(shape[i][j], end = " ")
        print()

def getMaxSize(lines, columns, shape):
    line, column = 0, 0
    for i in range(lines):
        for j in range(columns):
            if shape[i][j] != 0:
                try:
                    line = max(line, i)
                    column = max(column, j)
                except IndexError:
                    pass
    return line+1, column+1
            
def testBoard(lines, columns, shape):
    for i in range(lines):
        for j in range(columns):
            if shape[i][j] == 0:
                return False
    return True

def geometric(lines, columns, noShapes, shapes):
    trials = int(input("Maximum trials: "))
    board = np.zeros((lines, columns), dtype = int)
    for _ in range(trials):
        newBoard = copy.deepcopy(board)
        for shape in shapes:
            maxSize = getMaxSize(lines, columns, shape)
            maxLine = maxSize[0]
            maxColumn = maxSize[1]

            newLine = random.randint(0, lines - maxLine)
            newColumn = random.randint(0, columns - maxColumn)
            
            for i in range(maxLine):
                for j in range(maxColumn):
                    if shape[i][j] != 0:
                        if(newBoard[newLine + i][newColumn + j] != 0):
                            break
                        newBoard[newLine + i][newColumn + j] = shape[i][j]
                            
                    

        printMatrix(lines, columns, newBoard)
        if testBoard(lines, columns, newBoard):
            print('The solution was found!')
        else:
            print('The solution was not found!')
            
lines, columns, noShapes, shapes = readFromFile('geometric01.in')
geometric(lines, columns, noShapes, shapes)
