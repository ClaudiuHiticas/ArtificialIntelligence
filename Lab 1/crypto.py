import copy
import random
import string

def isPosible(words, symb, result, chipher):
    for word in words:
        if chipher[word[0]] == '0x0':
            return False
    if chipher[result[0]] == '0x0':
        return False

    words_aux = []
    for w in words:
        elems = []
        for letter in list(w):
            elems.append(chipher[letter][2]) 
        words_aux.append(''.join(elems))

    elems = []
    for letter in list(result):
        elems.append(chipher[letter][2]) 
    result_aux = ''.join(elems)

    words_ints = []
    for w in words_aux:
        words_ints.append(int(w,16))

    result_ints = int(result_aux, 16)
    # print("wa:",words_aux)
    # print("ra:", result_aux)
    # print("wi:", words_ints)
    # print("ri:", result_ints)

    if (symb == '+' and sum(words_ints) == result_ints) or \
            (symb == '-' and words_ints[0] - sum(words_ints[1:]) == result_ints):
        return True
    return False


def split(word): 
    return [char for char in word] 

def setElement(words, result):
    letters = set()
    for word in words:
        letters.update(split(word))
    letters.update(split(result))
    elements = dict.fromkeys(letters, '')
    return elements


def crypto(): 
    words = ['TAKE', 'A', 'CAKE']
    symb = '+'
    result = 'KATE'

    tries = int(input("Number of tries = "))

    elements = setElement(words, result)

    initTries = tries
    while tries:
        newElem = copy.deepcopy(elements)
        for key in newElem:
            newElem[key] = hex(random.randint(0, 15))
        print(initTries-tries+1, " : ",  newElem)
        tries -= 1

        if isPosible(words, symb, result, newElem):
            print('The solution was foud!')
        else:
            print('The solution was not found!')
crypto()