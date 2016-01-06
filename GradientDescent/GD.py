import sys
import csv
import numpy as np
import Features

# GD.py
# Author: Ayush

def readFile(filePath):
    with open(filePath, 'r') as destFile:
        dataIter = csv.reader(destFile, delimiter=',',quotechar='"')
        data = [data for data in dataIter]
    dataArray = np.asarray(data)                
    return dataArray

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

def parseArgs(args):
  """Parses arguments vector, looking for switches of the form -key {optional value}.
  For example:
	parseArgs([ 'template.py', '-i', '10', '-r', 'l1', '-s', '0.4', '-l', '0.5', '-f', '1' ]) = {'-i':'10', '-r':'l1', '-s:'0.4', '-l':'0.5', '-f':1 }"""
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

    maxIterations = 600 # the maximum number of iterations. should be a positive integer
    regularization = 'l2' # 'l1' or 'l2'
    stepSize = 0.05 # 0 < stepSize <= 1
    lmbd = 0.001 # 0 < lmbd <= 1
    featureSet = 1 # 1: original attribute, 2: pairs of attributes, 3: both

    if '-i' in args_map:
      maxIterations = int(args_map['-i'])
    if '-r' in args_map:
      regularization = args_map['-r']
    if '-s' in args_map:
      stepSize = float(args_map['-s'])
    if '-l' in args_map:
      lmbd = float(args_map['-l'])
    if '-f' in args_map:
      featureSet = int(args_map['-f'])

    assert maxIterations > 0
    assert regularization in ['l1', 'l2']
    assert stepSize > 0 and stepSize <= 1
    assert lmbd > 0 and lmbd <= 1
    assert featureSet in [1, 2, 3]
	
    return [maxIterations, regularization, stepSize, lmbd, featureSet]

# Extract label
def extractLabel(labels):
    vector = np.zeros(len(labels))
    for i in range(len(labels)):
        if labels[i] == '+':
            vector[i] = 1
        elif labels[i] == '-':
            vector[i] = -1
    return vector

def dotProduct(weights, vector):
    result = 0
    for i in vector:
        result = result + weights[i]
    return result

## Gradient Descent Algorithm
def GD(x, y, m, maxIterations, regularization, stepSize, lmbd, featureSet):

    n = Features.getLength(featureSet) + 1
    theta = np.zeros(n)

    for i in range(0, maxIterations):
        gradient = np.zeros(n)
        for j in range(m):
            if y[j]*dotProduct(theta, x[j]) <= 1:
                for item in x[j]:
                  gradient[item] = gradient[item] + y[j]
        bias = gradient[0]
        if regularization == 'l2':
          gradient = gradient - lmbd*theta
        elif regularization == 'l1':
          gradient = gradient - lmbd*np.sign(theta)
        gradient[0] = bias
        theta = theta + stepSize*gradient
        print "Theta: " + str(theta)

    return theta

def predict(x, m, theta, featureSet):
    y = np.ones(m)
    for i in range(m):
        result = dotProduct(theta, x[i])
        if result < 0:
            y[i] = -1
    return y

# main
# ----
# The main program loop
# You should modify this function to run your experiments

def main():
    arguments = validateInput(sys.argv)
    maxIterations, regularization, stepSize, lmbd, featureSet = arguments
    print maxIterations, regularization, stepSize, lmbd, featureSet

    trainData = readFile('train.csv')
    validationData = readFile('validation.csv')
    testData = readFile('test.csv')

    trainSize = trainData.shape[0]
    validationSize = validationData.shape[0]
    testSize = testData.shape[0]

    print "Number of training examples: " + str(trainSize)

    # Extract Features
    Features.extractFeatures(trainData[:,0], featureSet)
    print "Extracted Features:"
    if featureSet == 1 or featureSet == 3:
        print "Unigram: " + str(Features.getLength(1))
    if featureSet == 2 or featureSet == 3:
        print "Bigram: " + str(Features.getLength(2))

    # Construct Input Matrices X
    xTrain = Features.getMatrix(trainData[:,0], featureSet)
    print "Train Matrix built"

    xValidation = Features.getMatrix(validationData[:,0], featureSet)
    print "Validation Matrix built"

    xTest = Features.getMatrix(testData[:,0], featureSet)
    print "Test Matrix built"
    
    yTrain = extractLabel(trainData[:,1])
    yVailidation = extractLabel(validationData[:,1])
    yTest = extractLabel(testData[:,1])

    # Train the model
    theta = GD(xTrain, yTrain, trainSize, maxIterations, regularization, stepSize, lmbd, featureSet)
    print "Final Theta: " + str(theta)

    # Classify
    trainResult = predict(xTrain, trainSize, theta, featureSet)
    print "Train Result: " + str(trainResult)
    validationResult = predict(xValidation, validationSize, theta, featureSet)
    print "Validation Result: " + str(validationResult)
    testResult = predict(xTest, testSize, theta, featureSet)
    print "Test Result: " + str(testResult)

    # Performance
    print "\nPerformance on training data:"
    performance(trainResult, trainData[:,1])
    print "\nPerformance on validation data:"
    performance(validationResult, validationData[:,1])
    print "\nPerformance on test data:"
    performance(testResult, testData[:,1])
    

if __name__ == '__main__':
    main()
