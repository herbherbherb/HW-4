import numpy as np
import numpy.linalg as la
from copy import copy, deepcopy
import time, random
from collections import defaultdict as setdefault

def main():
#=======initialize================
	start_time = time.time()

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
		content_training[i] = [1 if(x == '#' or x == '+') else 0 for x in content_training[i][:-1]]


	with open('testimages') as f:
		content_testing = f.readlines()

	for i in range(len(content_testing)):
		content_testing[i] = [1 if(x == '#' or x == '+') else 0 for x in content_testing[i][:-1]]

	end = time.time()
	print("Reading Time: ", end - start_time)
#==================Testing==================
	start_time = time.time()
	counter = 0
	correct_predict = 0
	k = 4
	for i in range(int(len(content_testing)/28)):
		counter += 1
		curr_img = content_testing[i*28:(i+1)*28][0:28]
		result = testing(curr_img, content_training, traininglabels, k)
		if result == testlabels[i]:
			correct_predict += 1
		if counter%10 == 0:
			curr_acc = (correct_predict/counter)*100
			print(counter, ": ", "{0:.2f}".format(100*correct_predict/counter), "%", end = "  ")
			print()
			
	end = time.time()
	print("Testing Time: ", end - start_time)

def testing(img, content, traininglabels, k):
	predict = [0 for i in range(10)]
	result = []
	for i in range(int(len(content)/28)):
		label = traininglabels[i]
		train_img = content[i*28:(i+1)*28][0:28]
		difference = np.asarray(train_img) - np.asarray(img)
		difference = la.norm(difference, 2)
		result.extend([(label, difference)])

	result.sort(key=lambda tup:tup[1])
	new_result = [x[0] for x in result]
	new_result = new_result[0: k+1]
	counts = np.bincount(new_result)
	return np.argmax(counts)
main()