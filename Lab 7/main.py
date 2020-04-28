import numpy
from copy import deepcopy

class LeastSquare:
    def __init__(self):
        self.data = None
        self.result = None

    def readFromFile(self, fileName):
        f = open(fileName)
        data = []
        result = []
        for row in f:
            row = row.split(' ')
            vec = [1]
            for idx in range(len(row)-1):
                vec.append(float(row[idx]))
            result.append(float(row[-1]))
            data.append(vec)
        f.close()
        self.data = data
        self.result = result        


    def comupteResult(self, theta, data):
        sum = 0
        n = len(theta)
        for idx in range(n):
            sum += theta[idx] * data[idx]
        return sum

    def solve(self, n):
        trainingData = self.data[:n]
        testingData = self.data[n:]
        trainingResult = deepcopy(self.result[:n])
        testingResult = deepcopy(self.result[n:])

        #theta = ((At * A)^-1) * At * y;  y = result

        A = numpy.array(trainingData)       #our matrix
        At = A.transpose()                  #transpose of matrix
       
        #Algorithm
        Mult = numpy.matmul(At, A)          #(At * A)
        Inv = numpy.linalg.inv(Mult)        #(At * A)^-1
        Result = numpy.matmul(Inv, At)      #((At * A)^-1) * At
        
        Ay = numpy.array(trainingResult)    #the matrix of results
        Theta = numpy.matmul(Result, Ay)    #((At * A)^-1) * At * y

        #find the error
        error = 0
        for idx in range(len(testingData)):
            currentError = 0
            currentResult = self.comupteResult(Theta, testingData[idx])
            currentError = (testingResult[idx] - currentResult) ** 2
            error += currentError
            print('{:<4} Our result: {:<10} Official result: {:<10} Difference: {:<10}'.format(str(idx), str(round(currentResult,2)), str(testingResult[idx]), str(round(currentError,4))))
        
        print('Final error: ', str(round(error,6)))

def main():
    ls = LeastSquare()
    ls.readFromFile('data.csv')
    n = int(input("How many training data do you want to use? "))

    ls.solve(n)

main()
        

