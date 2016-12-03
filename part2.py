import numpy as np
from copy import copy, deepcopy
import time, random
from collections import defaultdict as setdefault
# import matplotlib.pyplot as plt

def main():
#=======initialize================
	start_time = time.time()
	dic = {}
	traininglabels = []
	testlabels = []
	testresult = []

#==================Reading==================
	with open('traininglabels') as f:
		content = f.readlines()
	traininglabels = [int(x.strip('\n')) for x in content]

	with open('testlabels') as f:
		content = f.readlines()
	testlabels = [int(x.strip('\n')) for x in content]

	with open('trainingimages') as f:
		content_training = f.readlines()
	
	for i in range(len(content_training)):
		content_training[i] = [1 if(x == '#' or x == '+') else 0 for x in content_training[i]]

	with open('testimages') as f:
		content_testing = f.readlines()

	for i in range(len(content_testing)):
		content_testing[i] = [1 if(x == '#' or x == '+') else 0 for x in content_testing[i]]

	end = time.time()
	print("Reading Time: ", end - start_time)
#==================Training==================
	start_time = time.time()
	for row in range(28):
		for col in range(28):
			loc = (row, col)
			if loc not in dic:
				dic[loc] = np.random.uniform(low=-1.0, high=1.0, size=(10,))
				# dic[loc] = np.zeros(10)
	
	epoch = 3
	learning_rate = 1000/(1000 + epoch)
	for times in range(epoch):
		training(content_training, traininglabels, dic, learning_rate)
		print("Epoch: ", times+1)
	end = time.time()
	print("Training Time: ", end - start_time)
#==================Testing==================
	testresult = testing(content_testing, testlabels, dic)
	correct_predicted = len([i for i, j in zip(testlabels, testresult) if i == j])
	accuracy_rate = (correct_predicted/len(testlabels))*100
	print("Accuracy Rate: ", accuracy_rate)

#==================Report==================
	report(testlabels, testresult)

def report(testlabels, testresult):
	digit = np.zeros((10))
	correct_digit = np.zeros((10))
	confusion = np.zeros((11, 11))
	confusion[0, 1:] = np.arange(10)
	confusion[1:, 0] = np.arange(10)

	for i in range(len(testresult)):
		digit[testlabels[i]] += 1
		if testresult[i] == testlabels[i]:
			correct_digit[testlabels[i]] += 1

		confusion[testlabels[i]+1][testresult[i]+1] += 1

	print("Classification Rate for Digits 0 ~ 9:")
	for i in range(10):
		print(i, ": ", "{0:.2f}".format(100*correct_digit[i]/digit[i]), "%", end = "  ")
		if i == 3 or i == 6:
			print()
	print()
	confusion_mtx(testresult, testlabels, confusion, digit)

def confusion_mtx(testresult, testlabels, confusion, digit):
	for i in range(1, 11):
		confusion[i, 1:] /= digit[i-1]
	print()
	np.set_printoptions(precision=2)
	print(confusion)

def testing(content, testlabels, dic):
	result = []
	for i in range(int(len(content)/28)):
		predict = [0 for i in range(10)]
		for row in range(i*28, (i+1)*28):
			for col in range(28):
				loc = (row%28, col)
				cur_char = content[row][col]
				for j in range(10):
					predict[j] += dic[loc][j] * cur_char
		predict_max = predict.index(max(predict))
		result.extend([predict_max])
	return result

def training(content, traininglabels, dic, rate):
	for i in range(int(len(content)/28)):
		cur_label = traininglabels[i]
		predict = [0 for i in range(10)]
		for row in range(i*28, (i+1)*28):
			for col in range(28):
				loc = (row%28, col)
				cur_char = content[row][col]
				for j in range(10):
					predict[j] += dic[loc][j] * cur_char
		predict_max = predict.index(max(predict))
		if predict_max == cur_label:
			continue
		else:
			for row in range(i*28, (i+1)*28):
				for col in range(28):
					loc = (row%28, col)
					cur_char = content[row][col]
					dic[loc][predict_max] -= rate * cur_char
					dic[loc][cur_label] += rate * cur_char	
main()