import sys
import os.path
import numpy as np
import pandas as pd
import math
from random import *

class neuralNetwork:

	def __init__ ( self, inputnodes, hiddennodes,outputnodes, learningrate):
		#set number of nodes in each input, hidden, output layer
		self.inodes = inputnodes
		self.hnodes = hiddennodes
		self.onodes = outputnodes

		#learning rate
		self.lr =learningrate
		pass

	def train(self, inputs, l2w, l3w, label):
		input2hidden = np.dot(l2w , inputs)
		print(str("input and weights dot: "))
		#print(input2hidden)
		hiddenVal = self.sigmoid(input2hidden)
		#print(str("Sigmoid layer 2"))
		#print(hiddenVal)

		hiddentoOut = np.dot(l3w, hiddenVal)
		#print(str("hiddenOUt dot: "))
		#print(hiddentoOut)
		layer3Val = self.sigmoid(hiddentoOut)
		
		print(layer3Val)
		delta = []
		for i, j in zip(label, layer3Val):
			delta.append(i - j)
		pass

	def query():
		pass

	def getValues(self):
		filename ="mnist_train.csv"
		#pd.read_csv(filename)
		values = [];
		if not os.path.isfile(filename):
		    print('File does not exist.')
		else:  
		    with open(filename) as textFile:
		    	lines = [line.split(",") for line in textFile]
		    	values = lines;
		    	#print(int(lines[1][10]))
		#randomly generated weights

		#pop the label
		# p = 0
		# while p < len(values):
		# 	values[p].pop(0)
		# 	p = p + 1
		values = np.int_(values)
		return values;


	def randomValues(self, x, y):
		fullWeights =[]
		i = 0
		#The two matrices have to be dot product compatible
		#Cols of matrix 1 should be equal to rows of matrix 2
		while i < y:
			fullWeights.append(np.random.uniform(low=-1.0, high=1.0, size=x))
			#print(fullWeights)
			i = i + 1
		return fullWeights


	def sigmoid(self, x):
		sigmoidArr = []
		y = 0
		while y < len(x):
			#print(str("Before Sigmoid: ") + str(weightxvalues[y]))
			sigmoidArr.append(1 / (1 + math.exp(-x[y]))) 
			#print(str(sigmoidArr[y]))
			y = y + 1
		return sigmoidArr


	def normalizeVales(self, value):
		minVal = np.amin(value)
		maxVal = np.amax(value)
		y = []
		i = 0
		while i < len(value):
			y.append((value[i] - minVal) / (maxVal - minVal))
			i = i + 1
		return y

	def buildLabArr(self, label):
		myList=[]
		for i in range(10):
			if i == label:
				myList.append(.99)
			else:
				myList.append(.1)
		print(myList)
		return myList
	# print("after normalizing")
	# print(valArr)
def main():
	nn = neuralNetwork("", "","","" )
	#Read vectors from file: first value is label and rest is vector in row 
	#image data converted to RGB value its a 28x28 matrix - then flattened 1D vector 784 pixels for each label 
	# input layer will ahve 784 nodes
	#seed forward , input weight forward passing 
	#weights = np.random.uniform(low=.1, high=.9, size=784)

	#generate random weihts tice once for in to hidden and hiden out
	values =  nn.getValues()
	layer2weights= nn.randomValues(784, 100)
	layer3weights= nn.randomValues(100, 10)
	i = 0
	while i < 1:
		normalVals = nn.normalizeVales(values[i][1:])
		#print(normalVals)
		label = values[i][0]
		labArr = nn.buildLabArr(label)
		#inputVal = values[0][1:]
		#print(values[0][1:])
		#print(fullWeights)
		nn.train(normalVals, layer2weights, layer3weights, labArr) 
		i = i + 1



if __name__ == "__main__":
    main()






