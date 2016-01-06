# DecisionTree.py
# ---------------
# Ayush Parolia

# Imports
import numpy as np
import Tree
import math

def majorityLabel(values):
    total = values.sum()
    size = values.size

    if total <= size/2:
        return 0
    else:
        return 1

def sameLabel(values):
    total = values.sum()
    size = values.size

    if total == 0 or total == size:
        return True

    return False

def entropy(data, label):
    totalSize = data[:,label].size

    if totalSize == 0:
        return 0.0
    
    ones = data[:,label].sum()
    zeros = totalSize - ones

    p0 = float(zeros)/totalSize
    p1 = float(ones)/totalSize

    if p0 == 0 or p1 == 0:
        return 0.0
    
    return -(p0*math.log(p0,2) + p1*math.log(p1,2))

def getExamples(data, attribute, value):
    totalRows = data[:,1].size
    rowsToDelete = []
    
    for i in range(totalRows):
        if value != data[i,attribute]:
            rowsToDelete.append(i)

    data = np.delete(data, rowsToDelete, axis=0)

    return data

def gain(data, attr, label):
    totalSize = data[:,label].size
    subsetEntropy = 0.0

    for i in range(1,11):
        dataSubset = getExamples(data, attr, i)
        subsetEntropy += (float(dataSubset[:,label].size)/totalSize)*entropy(dataSubset, label)

    return entropy(data, label) - subsetEntropy

def chooseAttr(data, label, attributes):
    best = attributes[0]
    maxGain = 0

    for attr in attributes:
        newGain = gain(data, attr, label)
        if newGain > maxGain:
            maxGain = newGain
            best = attr

    return best

def makeTree(data, label, attributes, level, max_depth):
    level += 1
    default = majorityLabel(data[:,label])

    # Max depth check
    if max_depth == level:
        return Tree.Tree(default, level)

    if len(attributes) <= 0:
        return Tree.Tree(default, level)
    elif sameLabel(data[:,label]):
        return Tree.Tree(default, level)
    else:
        bestAttribute = chooseAttr(data, label, attributes)
        tree = Tree.Tree(bestAttribute, level)
        newAttributes = attributes
        newAttributes.remove(bestAttribute)

        for i in range(1,11):
            examples = getExamples(data, bestAttribute, i)
            if examples[:,1].size == 0:
                tree.addChild(Tree.Tree(default,level+1),i)
            else:
                tree.addChild(makeTree(examples, label, newAttributes, level, max_depth),i)

    return tree
