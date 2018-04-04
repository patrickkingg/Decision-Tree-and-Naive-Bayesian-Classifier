#done by Patrick Wang
import csv
from sys import argv
import numpy as np

def main():

	trainingData = argv[1]
	testData = argv[2]
	outputFile = argv[3]


	reader = csv.reader(open(trainingData), delimiter='\t')
	df = np.array([row for row in reader])

	bayes = NaiveBayesian()
	bayes.fit(df)

	reader = csv.reader(open(testData), delimiter='\t')
	dfTest = np.array([row for row in reader])

	xTest, yTest = dfTest[:,1:], dfTest[:, 0]
	score = bayes.scoring(xTest, yTest)

	print("naive bayesian accuracy rate {}%".format(score))
	with open(outputFile, 'w') as f:
		f.write("\nnaive bayesian accuracy rate {}%".format(score))
		f.write('\n'.join(bayes.prediction(xTest)))
		f.write("\nnaive bayesian accuracy rate {}%".format(score))

class NaiveBayesian(object):

	def __init__(self):
		self.aClass = None
		self.priors = {}
		self.table = []

	def prediction(self, xTest):
		result = []
		for x in xTest:
			result1 = {}
			for aClass in self.aClass:
				prob = 1
				for i in range(len(list(x))):
					prob =prob* self.table[i][x[i]][aClass]
				result1[aClass] = prob

			maxVal = max(result1.values())
			for k,v in result1.items():
				if v==maxVal:
					maxKey = [k][0]

			result.append(maxKey)

		return result

	def scoring(self, xTest, yTest):
		result = self.prediction(xTest)
		yTest = np.array(yTest)
		answer = sum(result == yTest) * 100.0
		answer = answer/ yTest.size
		return answer

	def fit(self, df):
		X, y = list(df[:, 1:]), df[:, 0]
		classesOther = np.unique(y)
		attributes=[]
		for i in zip(*X):
			attributes.append(list(np.unique(i)))

		priors = {}
		classes = np.unique(y)
		for aClass in classes:
			pri = sum(y == aClass) * 1.0 / y.shape[0]
			priors[aClass] = pri
		self.priors = priors

		table = []
		for i in range(1, len(attributes)+1):
			attribute = attributes[i-1]
			dict1 = dict()
			for a in attribute:
				dict2 = dict()
				for aClass in classesOther:
					dfClass = df[y==aClass, :]
					classNumber = dfClass.shape[0]
					attributeNumber = sum(dfClass[:, i] == a)
					dict2[aClass] = attributeNumber * 1.0 / classNumber
				dict1[a] = dict2

			table.append(dict1)

		self.table = table
		self.aClass = classesOther

if __name__ == '__main__': main()