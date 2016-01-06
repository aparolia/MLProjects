import numpy as np
import re

# Features.py
# Author: Ayush Parolia

prepositions = {'is', 'it', 'and', 'as', 'in', 'the', 'to', 'this', 'its', 'of', 'a', 'by'}
unigram = set()
bigram = set()

def getWords(text):
    rawList = re.findall(r"[\w']+", text)
    return set(rawList)-prepositions

def getBiWords(text):
    biWords = set()
    rawList = re.findall(r"[\w']+", text)
    for i in range(len(rawList)-1):
        biWords.add(rawList[i] + " " + rawList[i+1])
    return biWords

def extractFeatures(data, feature):
    global unigram
    global bigram
    for i in range(data.size):
        if feature == 1 or feature == 3:
            unigram = unigram.union(getWords(data[i]))
        if feature == 2 or feature == 3:
            bigram = bigram.union(getBiWords(data[i]))

def getVector(row, feature):
    global unigram
    global bigram
    vector = []
    index = 0
    vector.append(index)
    if feature == 1 or feature == 3:
        rowWords = getWords(row)
        for item in unigram:
            index += 1
            if item in rowWords:
                vector.append(index)
    if feature == 2 or feature == 3:
        rowBiWords = getBiWords(row)
        for item in bigram:
            index += 1
            if item in rowBiWords:
                vector.append(index)
    return vector

def getMatrix(data, feature):
    matrix = []
    m = len(data)
    for i in range(m):
        matrix.append(getVector(data[i], feature))
    return matrix

"""
def getMatrix(data, feature):
    global unigram
    global bigram

    # number of samples
    m = data.shape[0]
    n = 0
    
    if feature == 1:
        n = len(unigram) + 1
        matrix = np.zeros((m,n), bool)
    elif feature == 2:
        n = len(bigram) + 1
        matrix = np.zeros((m,n), bool)
    else:
        n = len(unigram) + len(bigram) + 1
        matrix = np.zeros((m,n), bool)

    for i in range(m):
        index = 0
        # Bias Term
        matrix[i,index] = 1
        if feature == 1 or feature == 3:
            rowWords = getWords(data[i])
            for item in unigram:
                index += 1
                if item in rowWords:
                    matrix[i,index] = 1
        if feature == 2 or feature == 3:
            rowBiWords = getBiWords(data[i])
            for item in bigram:
                index += 1
                if item in rowBiWords:
                    matrix[i,index] = 1
    return matrix
"""

def getLength(featureSet):
    global unigram
    global bigram
    uniLength = len(unigram)
    biLength = len(bigram)
    if featureSet == 1:
        return uniLength
    elif featureSet == 2:
        return biLength
    elif featureSet == 3:
        return uniLength + biLength
