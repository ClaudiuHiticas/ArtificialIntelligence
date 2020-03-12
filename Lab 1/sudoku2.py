#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 19:32:49 2020

@author: claudiuhiticas
"""
import math
import random

def posible(x, y, nr, n, grid):
    for i in range(0, n):
        if grid[i][y] == nr:
            return False
    for j in range(0, n):
        if grid[x][j] == nr:
            return False
        
    rad = int(math.sqrt(n))
    x0 = (x//rad)*rad
    y0 = (y//rad)*rad
    for i in range(0, rad):
        for j in range(0, rad):
            if grid[x0+i][y0+j] == nr:
                return False
    return True
    
def solve(n, grid, maxim):
    cnt = 0
    for x in range(0, n):
        for y in range(0, n):
            if grid[x][y] == 0:
                while grid[x][y] == 0:
                    if maxim > cnt:
                        cnt += 1
                        nr = random.randint(1,n)
                        if posible(x, y, nr, n, grid):
                            grid[x][y] = nr
                    else:
                        print("Final solution not found!")
                        return;
    print(cnt, "trials")

def readFromFile(file):
    fin = open(file,'r')
    grid=[]
    for line in fin.readlines():
        grid.append( [ int (x) for x in line.split(',') ] )
    return grid

def printMatrix(grid, n):
    for i in range(0, n):
        for j in range(0, n):
            print(grid[i][j], end=" ")
        print()


def sudoku():
    n = int(input("Size of sudoku (4 or 9) = "))
    maxim = int(input("Max. no. trials: "))
    
    if n == 9:
        grid = readFromFile("sudoku02.in")
        solve(n, grid, maxim)
        printMatrix(grid, n)
    elif n == 4:
        grid = readFromFile("sudoku01.in")
        solve(n, grid, maxim)
        printMatrix(grid, n)
    else:
       print("Just with 4 and 9")
    

sudoku()