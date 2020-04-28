from copy import deepcopy
import numpy
import random
from numpy.distutils.conv_template import header

def isNumeric(value):
    return isinstance(value, int) or isinstance(value, float)

class Question:
    def __init__(self, column, value):
        self._column = column
        self._value = value

    @property
    def column(self):
        return self._column

    @property
    def value(self):
        return self._value

    def match(self, example):
        val = example[self.column]
        if isNumeric(val):
            return val >= self.value
        else:
            return val == self.value

    def __repr__(self):
        condition = "=="
        if isNumeric(self.value):
            condition = ">="
        return "Is %s %s %s?" % (
            header[self.column], condition, str(self.value))


class Leaf:
    def __init__(self, rows, helper):
        self.predictions = helper.counter(rows)


class DecisionNode:
    def __init__(self, question, trueBranch, falseBranch):
        self._question = question
        self._trueBranch = trueBranch
        self._falseBranch = falseBranch

    @property
    def question(self):
        return self._question

    @property
    def true(self):
        return self._trueBranch

    @property
    def false(self):
        return self._falseBranch


class Helper:
    def __init__(self):
        pass

    def counter(self, rows):
        count = {}
        for row in rows:
            label = row[0]
            if label not in count:
                count[label] = 0
            count[label] += 1
        return count

    def readFromFile(self, fileName):
        data = []
        f = open(fileName)
        while True:
            row = f.readline()
            if not row:
                break
            data.append([row[0], int(row[2]), int(row[4]), int(row[6]), int(row[8])])
        return data

    def testingData(self, data):
        random.shuffle(data)
        return deepcopy(data[: int(0.8 * len(data))]), deepcopy(data[int(0.2 * len(data)):])


class Controller:
    def __init__(self, helper):
        self._helper = helper

    def buildTree(self, rows):
        gain, question = self.bestSplit(rows)
        if gain == 0:
            return Leaf(rows, self._helper)
        true, false = self.partition(rows, question)
        trueBranch = self.buildTree(true)
        falseBranch = self.buildTree(false)
        return DecisionNode(question, trueBranch, falseBranch)

    def classify(self, row, node):
        if isinstance(node, Leaf):
            return node.predictions
        return self.classify(row, node.true) if node.question.match(row) else  self.classify(row, node.false)

    def partition(self, rows, question):
        true, false = [], []
        for row in rows:
            if question.match(row):
                true.append(row)
            else:
                false.append(row)
        return true, false

    def gini(self, rows):
        count = self._helper.counter(rows)
        impurity = 1
        for label in count:
            probOfLbl = count[label] / float(len(rows))
            impurity = impurity - probOfLbl ** 2
        return impurity

    def infoGain(self, left, right, giniIdx):
        p = float(len(left)) / (len(left) + len(right))
        return giniIdx - p * self.gini(left) - (1 - p) * self.gini(right)

    def bestSplit(self, rows):
        bestGain = 0
        bestQuestion = None
        giniIdx = self.gini(rows)
        noFeatures = 5
        values = [1, 2, 3, 4, 5]
        for col in range(1, noFeatures):
            for val in values:
                quest = Question(col, val)
                true, false = self.partition(rows, quest)
                if len(true) == 0 or len(false) == 0:
                    continue
                gain = self.infoGain(true, false, giniIdx)
                if gain > bestGain:
                    bestGain = gain
                    bestQuestion = quest
        return bestGain, bestQuestion


def main():
    helper = Helper()
    data = helper.readFromFile("balance-scale.data")
    statist = []
    for i in range(100):
        train, test = helper.testingData(data)
        ctrl = Controller(helper)
        tree = ctrl.buildTree(train)
        done = 0
        total = len(test)
        for row in test:
            if row[0] == list(ctrl.classify(row, tree).keys())[0]:
                done = done + 1
        p = (done / total) * 100
        statist.append(p)
    print(numpy.mean(statist))

main()