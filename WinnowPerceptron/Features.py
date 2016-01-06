import numpy as np
import re

# Features.py
# -------
# Ayush Parolia

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
    index = -1
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
        
