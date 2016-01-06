# id3.py
# -------
# Ayush Parolia

#Import
import sys
import numpy as np
import DecisionTree

##Classify
#---------

def classify(decisionTree, examples):
    results = np.zeros(examples[:,1].size)
    for i in range(examples[:,1].size):
        node = decisionTree
        x = examples[i,:]
        while not node.isLeaf():
            #print node.value
            node = node.children[x[node.value]]
        results[i] = node.value
        #print "result"
        #print results[i]
    return results


##Learn
#-------
def learn(dataset, pruneFlag, maxDepth):
    tree = DecisionTree.makeTree(dataset, 9, [0,1,2,3,4,5,6,7,8], -1, maxDepth)
    return tree

##Performance
#------------
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
        if prediction[i] == 0 and actual[i] == 0:
            true_negative += 1
            match += 1
        elif prediction[i] == 1 and actual[i] == 1:
            true_positive += 1
            match += 1
        elif prediction[i] == 1 and actual[i] == 0:
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
    parseArgs([ 'main.py', '-p', 5 ]) = {'-p':5 }"""
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
    valSetSize = 0
    pruneFlag = False
    maxDepth = -1
    if '-p' in args_map:
      pruneFlag = True
      valSetSize = int(args_map['-p'])
    if '-d' in args_map:
      maxDepth = int(args_map['-d'])
    return [pruneFlag, valSetSize, maxDepth]

def main():
    arguments = validateInput(sys.argv)
    
    pruneFlag, valSetSize, maxDepth = arguments

    # Read in the data file
    train_data = np.loadtxt('training.csv', dtype=int)
    validation_data = np.loadtxt('validating.csv', dtype=int)
    test_data = np.loadtxt('testing.csv', dtype=int)

    print "Prune flag = " + str(pruneFlag)
    print "Maximum Depth = " + str(maxDepth)

    # Learn
    tree = learn(train_data, pruneFlag, maxDepth)

    # Print decision tree
    print "Decision Tree"
    print tree

    # Classify
    train_result = classify(tree, train_data[:,0:9])
    val_result = classify(tree, validation_data[:,0:9])
    test_result = classify(tree, test_data[:,0:9])

    # Print results
    print "Training set labels:"
    print train_result
    print "Validation set labels"
    print val_result
    print "Test set labels"
    print test_result

    # Measure performnce
    print "\nPerformance on training data:"
    performance(train_result, train_data[:,9])
    print "\nPerformance on validation data:"
    performance(val_result, validation_data[:,9])
    print "\nPerformance on test data:"
    performance(test_result, test_data[:,9])


if __name__ == '__main__':
    main()



