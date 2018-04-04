#done by PAtrick Wang
import csv
from sys import argv
import numpy as np

def main():

    trainingData = argv[1]
    testData = argv[2]
    outputFile = argv[3]
    reader = csv.reader(open(trainingData), delimiter='\t')
    dfTrain = np.array([row for row in reader])

    x = list(dfTrain[:, 1:])
    classes1 = [None]
    for i in zip(*x):
        classes1.append(list(np.unique(i)))

    root = build_decision_tree(dfTrain, list(range(1, 21)), classes1)
    reader = csv.reader(open(testData), delimiter='\t')
    dfTest = np.array([row for row in reader])
    xTest, yTest = dfTest[:, 1:], dfTest[:, 0]

    result=[]
    for x in xTest:
        leaf = root
        while leaf.crit:
            key = x[leaf.crit-1]
            leaf = leaf.kids[key]
        result.append(leaf.value)
    result = np.array(result)
    yTest = np.array(yTest)

    percentage = sum(result == yTest) * 100.0 / yTest.size
    print("C4.5 accuracy {} %".format(percentage))
    with open(outputFile, 'w') as f:
        f.write("\nC4.5 accuracy {}%".format(percentage))
        f.write('\n'.join(result))
        f.write("\nC4.5 accuracy {}%".format(percentage))


class Leaf(object):
    def __init__(self):
        self.value = None
        self.kids = {}
        self.crit = None

def gainRatio(df, index):
    y = list(df[:, 0])
    classes = np.unique(y)
    probs=[]
    for aClass in classes:
        probs.append(y.count(aClass) * 1.0 / len(y))

    info1 = -sum([pr * np.log2(pr) for pr in probs])

    classes0 = np.unique(df[:, index])
    dfBranches = {aClass:[] for aClass in classes0}
    for each in df:
        aClass = each[index]
        dfBranches[aClass].append(each)
    info2 = 0
    for key in dfBranches:
        freq = len(dfBranches[key]) * 1.0 / df.shape[0]
        y1 = list(np.array(dfBranches[key])[:, 0])
        classes1 = np.unique(y1)
        probs1 = []
        for aClass in classes1:
            probs1.append(y1.count(aClass) * 1.0 / len(y1))

        answer = -sum([prob * np.log2(prob) for prob in probs1])
        info2 += freq * answer


    splitInfo=0
    for i in dfBranches:
        frequency = len(dfBranches[i]) * 1.0/(df.shape[0])
        splitInfo = splitInfo + (-frequency*np.log2(frequency))

    if splitInfo == 0:
        return 0
    else:
        return (info1 - info2) / splitInfo

def build_decision_tree(df, attributes, theClasses):
    leaf = Leaf()
    y = df[:, 0]
    if np.unique(y).size == 1:
        leaf.value = np.unique(y)[0]
        return leaf
    value = max(attributes, key=lambda x: gainRatio(df, x))
    leaf.crit = value
    attributes.pop(value - 1)

    classes = np.unique(df[:,value])
    dfBranch = {aClass: [] for aClass in classes}

    for row in df:
        class1 = row[value]
        dfBranch[class1].append(row)


    theClass = theClasses[value]

    for key in theClass:
        if key not in dfBranch:
            leaf = Leaf()
            y1=list(df[:, 0])
            leaf.value = max([i for i in np.unique(y1)], key=y1.count)

            leaf.kids[key] = leaf
        else:
            leaf.kids[key] = build_decision_tree(
                np.array(dfBranch[key]),
                attributes,
                theClasses)

    return leaf

if __name__ == '__main__':
    main()