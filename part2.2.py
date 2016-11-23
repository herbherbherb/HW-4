import numpy as np
from copy import copy, deepcopy
import time
from collections import defaultdict as setdefault
# import matplotlib.pyplot as plt

def main():
#=======initialize================
	start_time = time.time()

	traininglabels = []
	testlabels = []
	testresult = []

#========Reading==================
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
	print("Training Time: ", end - start_time)
#===========Testing================================================
	for i in range(len(testlabels)):
		

main()