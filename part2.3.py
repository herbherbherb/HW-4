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
	dic = {}
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
#==================Training==================
	start_time = time.time()
	counter = 0
	correct_predict = 0
	k = 7

	coord = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), \
					(1, -1), (1, 0), (1, 1)]
	random.shuffle(coord)
	coord = (coord)[0:k]
	training(content_training, dic, coord, k)
	end = time.time()
	print("Training Time: ", end - start_time)

#==================Testing==================
	start_time = time.time()
	training_times = int(len(content_training)/28)
	for i in range(int(len(content_testing)/28)):
		counter += 1
		curr_img = content_testing[i*28:(i+1)*28][0:28]
		result = testing(curr_img, training_times, traininglabels, dic, coord, k)
		if result == testlabels[i]:
			correct_predict += 1
		if counter%10 == 0:
			curr_acc = (correct_predict/counter)*100
			print(counter, ": ", "{0:.2f}".format(100*correct_predict/counter), "%", end = "  ")
			print()
			
	end = time.time()
	print("Testing Time: ", end - start_time)

def training(content, dic, curr_coord, k):
	for i in range(int(len(content)/28)):
		if i not in dic:
			dic.setdefault(i, {})
		train_img = content[i*28:(i+1)*28][0:28]
		for row in range(2, 26):
			for col in range(2, 26):
				cur_location = (row, col)
				train_vec = np.zeros(k)
				for loc in range(len(curr_coord)):
					cur_row = row + curr_coord[loc][0]
					cur_col = col + curr_coord[loc][1]
					train_vec[loc] = train_img[cur_row][cur_col]
				if cur_location not in dic[i]:
					dic[i][cur_location] = train_vec
				else:
					dic[i][cur_location] = train_vec


def testing(img, times, traininglabels, dic, curr_coord, k):
	predict = [0 for i in range(10)]
	for i in range(times):
		label = traininglabels[i]
		difference = 0
		for row in range(2, 26):
			for col in range(2, 26):
				cur_location = (row, col)
				curr_vec = np.zeros(k)
				train_vec = dic[i][cur_location]
				for loc in range(len(curr_coord)):
					cur_row = row + curr_coord[loc][0]
					cur_col = col + curr_coord[loc][1]
					curr_vec[loc] = img[cur_row][cur_col]

				difference += la.norm(train_vec - curr_vec, 2)
		predict[label] += difference
	return predict.index(min(predict))
main()