import sys
import csv
import numpy as np
import Features

# main.py
# -------
# Ayush Parolia

def readFile(filePath):
    with open(filePath, 'r') as destFile:
        dataIter = csv.reader(destFile, delimiter=',',quotechar='"')
        data = [data for data in dataIter]
    dataArray = np.asarray(data)                
    return dataArray

def dotProduct(weights, vector):
    result = 0
    for i in vector:
        result = result + weights[0,i]
    return result

##predict a single example
def predict_one(weights, inputVector, threshold):
    result = dotProduct(weights, inputVector)
    if result >= threshold:
        return 1
    else:
        return -1

def classify(data, weights, featureSet, algorithm):
    length = Features.getLength(featureSet)
    results = np.zeros(data.shape[0])
    for i in range(data.shape[0]):
        if algorithm == 1:
            vector = Features.getVector(data[i,0], featureSet)
            vector.append(length)
            results[i] = predict_one(weights, vector, 0)
        else:
            vector = Features.getVector(data[i,0], featureSet)
            results[i] = predict_one(weights, vector, length)
        
    return results    

##Perceptron
#-----------
def perceptron(data, maxIterations, featureSet):
    length = Features.getLength(featureSet)
    rate = 0.1
    weights = np.zeros((1,length+1))
    i = 0
    while i < maxIterations:
        for j in range(data.shape[0]):
            #print weights
            vector = Features.getVector(data[j,0], featureSet)
            vector.append(length)
            sign = predict_one(weights, vector, 0)
            if (data[j,1] == '+' and sign == -1) or (data[j,1] == '-' and sign == 1):
                for index in vector:
                    weights[0,index] = weights[0,index] - rate*sign
        i += 1
    return weights 


##Winnow
#-------
def winnow(data, maxIterations, featureSet):
    length = Features.getLength(featureSet)
    threshold = length
    weights = np.ones((1,length))
    i = 0
    while i < maxIterations:
        for j in range(data.shape[0]):
            #print weights
            vector = Features.getVector(data[j,0], featureSet)
            sign = predict_one(weights, vector, threshold)
            if data[j,1] == '+' and sign == -1:
                for index in vector:
                    weights[0,index] = weights[0,index]*2
            elif data[j,1] == '-' and sign == 1:
                for index in vector:
                    weights[0,index] = weights[0,index]/2
        i += 1
    return weights

def performance(prediction, actual):
    total = actual.size
    match = 0.0
    false_positive = 0.0
    false_negative = 0.0
    true_positive = 0.0
    true_negative = 0.0
    precision = 0.0
    recall = 0.0

    if total == 0:
        return
    
    for i in range(total):
        if prediction[i] == -1 and actual[i] == '-':
            true_negative += 1
            match += 1
        elif prediction[i] == 1 and actual[i] == '+':
            true_positive += 1
            match += 1
        elif prediction[i] == 1 and actual[i] == '-':
            false_positive += 1
        else:
            false_negative += 1

    print "Accuracy = " + str(match/total)
    if (true_positive + false_positive) != 0:
        precision = true_positive/(true_positive + false_positive)
        print "Precision = " + str(precision)
    if (true_positive + false_negative) != 0:
        recall = true_positive/(true_positive + false_negative)
        print "Recall = " + str(recall)
    if (precision+recall) != 0:
        print "F1 score = " + str(2*precision*recall/(precision+recall))

# main
# ----
# The main program loop
# You should modify this function to run your experiments

def parseArgs(args):
  """Parses arguments vector, looking for switches of the form -key {optional value}.
  For example:
	parseArgs([ 'template.py', '-a', 1, '-i', 10, '-f', 1 ]) = {'-t':1, '-i':10, '-f':1 }"""
  args_map = {}
  curkey = None
  for i in xrange(1, len(args)):
    if args[i][0] == '-':
      args_map[args[i]] = True
      curkey = args[i]
    else:
      assert curkey
      args_map[curkey] = args[i]
      curkey = None
  return args_map

def validateInput(args):
    args_map = parseArgs(args)

    algorithm = 1 # 1: perceptron, 2: winnow
    maxIterations = 10 # the maximum number of iterations. should be a positive integer
    featureSet = 1 # 1: original attribute, 2: pairs of attributes, 3: both

    if '-a' in args_map:
      algorithm = int(args_map['-a'])
    if '-i' in args_map:
      maxIterations = int(args_map['-i'])
    if '-f' in args_map:
      featureSet = int(args_map['-f'])

    assert algorithm in [1, 2]
    assert maxIterations > 0
    assert featureSet in [1, 2, 3]
	
    return [algorithm, maxIterations, featureSet]

def main():
    arguments = validateInput(sys.argv)
    algorithm, maxIterations, featureSet = arguments
    print algorithm, maxIterations, featureSet

    # ====================================
    # WRITE CODE FOR YOUR EXPERIMENTS HERE
    # ====================================

    trainData = readFile('train.csv')
    validationData = readFile('validation.csv')
    testData = readFile('test.csv')

    # Extract features
    Features.extractFeatures(trainData[:,0], featureSet)

    length = Features.getLength(featureSet)

    # Learn (Get weight vector)
    if algorithm == 1:
        weights = perceptron(trainData, maxIterations, featureSet)
    else:
        weights = winnow(trainData, maxIterations, featureSet)

    # Classify
    trainResult = classify(trainData, weights, featureSet, algorithm)
    validationResult = classify(validationData, weights, featureSet, algorithm)
    testResult = classify(testData, weights, featureSet, algorithm)

    # Performance
    print "\nPerformance on training data:"
    performance(trainResult, trainData[:,1])
    print "\nPerformance on validation data:"
    performance(validationResult, validationData[:,1])
    print "\nPerformance on test data:"
    performance(testResult, testData[:,1])

if __name__ == '__main__':
    main()
