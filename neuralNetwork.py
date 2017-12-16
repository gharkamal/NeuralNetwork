import sys
import os.path
import numpy as np
import pandas as pd
import math
from datetime import datetime
from numba import jit
from random import *

class neuralNetwork:

	def __init__ ( self, inputnodes, hiddennodes,outputnodes, learningrate):
		#set number of nodes in each input, hidden, output layer
		self.inodes = inputnodes
		self.hnodes = hiddennodes
		self.onodes = outputnodes
		np.random.seed(1)
		#learning rate
		self.learningrate =learningrate
		self.layer2weights= self.randomValues(inputnodes, hiddennodes)  #set the weights for the input -> hidden layer
		self.layer3weights= self.randomValues(hiddennodes, outputnodes) 	 #set the weights for the hidden -> output layer

		pass

	def train(self, inputs, label):
		#The two matrices have to be dot product compatible
		#Cols of matrix 1 should be equal to rows of matrix 2
		input2hidden = np.dot(self.layer2weights , inputs) #input layer and first set of weights 
		#print(str("input and weights dot: "))
		#print(np.asarray(input2hidden).shape)
		hiddenVal = self.sigmoid(input2hidden) #value of hidden layer
		#print(str("Sigmoid layer 2"))
		#print(hiddenVal)

		hiddentoOut = np.dot(self.layer3weights, hiddenVal) #hidden layer and 2nd set of weights
		# print(np.asarray((hiddentoOut)))
		#print(str("hiddenOUt dot: "))
		#print(hiddentoOut)
		layer3Val = self.sigmoid(hiddentoOut) # output
		#print(label)
		#print(layer3Val)
		#delta is the error (label and ouput value is used)
		layer3Error = np.asarray(label) - np.asarray(layer3Val)
		# for i, j in zip(label, layer3Val):
		# 	layer3Error.append(i - j)

		# layer3Error = np.asarray(layer3Error)
		hiddenVal = np.asarray(hiddenVal)
		layer3Val = np.asarray(layer3Val)

		# hiddenLayError = []
		# for i, j in zip(label, hiddenVal):
		# 	hiddenLayError.append(i - j)
		#print(layer3Error)
		# print(np.asarray(self.layer3weights).shape)
		# print(np.asarray(layer3Error).shape)
		hiddenErrordot = np.dot(np.asarray(self.layer3weights).T, layer3Error) 
		#get the error in the hidden layer by dot product of 2nd set of weights and the error we get from label array - sigmoid values of layer 3

		# hiddendot = np.dot(np.asarray(self.layer2weights).T, np.asarray(hiddenLayError).reshape(10,1))
		#errorDelta = abs((sum(delta))/(len(delta)))
		#print(errorDelta)
		#Start of back propagation 
		backPropLayer3 =  layer3Error * (layer3Val * (1.0 - layer3Val))

		backPropHidden =  hiddenErrordot * (hiddenVal * (1.0 - hiddenVal))	
		# print(backPropLayer3.shape)
		# print(hiddenVal.shape)
		#
		self.layer3weights += self.learningrate * np.dot(backPropLayer3.reshape(10,1), hiddenVal.reshape(1,100) )
		self.layer2weights += self.learningrate * np.dot(backPropHidden.reshape(100,1), np.asarray(inputs).reshape(1,784) )

		#print(backPropLayer3)
		#print(backPropHidden)	
		# print(backProp1)
		# self.layer3weights = 
		#print(len(delta))
		#print(delta)

		pass

	def query(self, inputs):
		input2hidden = np.dot(self.layer2weights , inputs) #input layer and first set of weights 
		hiddenVal = self.sigmoid(input2hidden) #value of hidden layer


		hiddentoOut = np.dot(self.layer3weights, hiddenVal) #hidden layer and 2nd set of weights
		layer3Val = self.sigmoid(hiddentoOut) # output
		return layer3Val


	def getValues(self, testData):
		if testData:
			filename ="data/mnist_test.csv"
			values = []; #all the values are returned. File is parsed and split
			if not os.path.isfile(filename):
			    print('File does not exist.')
			else:  
			    with open(filename) as textFile:
			    	lines = [line.split(",") for line in textFile]
			    	values = lines;
			values = np.int_(values)
			return values;
		else:
			filename ="data/mnist_train.csv"
			values = []; #all the values are returned. File is parsed and split
			if not os.path.isfile(filename):
			    print('File does not exist.')
			else:  
			    with open(filename) as textFile:
			    	lines = [line.split(",") for line in textFile]
			    	values = lines;
			values = np.int_(values)
			return values;

	#build random values using x and y for matrix
	def randomValues(self, x, y):
		fullWeights =[]
		i = 0
		while i < y:
			fullWeights.append(np.random.uniform(low=-1.0, high=1.0, size=x))
			i = i + 1
		return fullWeights

	#sigmoid function
	def sigmoid(self, x):
		sigmoidArr = []
		y = 0
		#for every value in the array after dot product 
		while y < len(x):
			sigmoidArr.append(1 / (1 + math.exp(-x[y]))) 
			y = y + 1
		return sigmoidArr #return array


	#normalize the array of values to be able to do functions
	def normalizeVales(self, value):
		minVal = np.amin(value)
		maxVal = np.amax(value)
		y = []
		i = 0
		while i < len(value):
			y.append((value[i] - minVal) / (maxVal - minVal))
			i = i + 1
		return y

	#build the array of the label for 10 values where label index is set to .99 rest is .1
	def buildLabArr(self, label):
		myList=[]
		for i in range(10):
			if i == label:
				myList.append(.99)
			else:
				myList.append(.1)
		#print(myList)
		return myList
@jit
def main():
	startTime = datetime.now()
	inputnode = 784
	hiddennodes = 100 
	outputnodes = 10
	learningrate = .3
	nn = neuralNetwork(inputnode,hiddennodes,outputnodes, learningrate )  #initialize neural network
	#Read vectors from file: first value is label and rest is vector in row 
	#image data converted to RGB value its a 28x28 matrix - then flattened 1D vector 784 pixels for each label 
	# input layer will ahve 784 nodes
	#seed forward , input weight forward passing 

	#generate random weihts tice once for in to hidden and hiden out
	trainValues =  nn.getValues(False)
	testValues = nn.getValues(True)

	epoch = 3
	#set epoch to run the training
	for y in range(epoch):
		i = 0
		while i < len(trainValues):
			normalVals = nn.normalizeVales(trainValues[i][1:])
			label = trainValues[i][0]
			labArr = nn.buildLabArr(label)
			nn.train(normalVals, labArr) 
			i = i + 1
	y = 0
	count = 0
	#query data to get the accuracy
	while y < len(testValues): #run for size of test data
		normalValsTest = nn.normalizeVales(testValues[y][1:])
		labelTest = testValues[y][0]
		prediction  = nn.query(normalValsTest) 
		if np.argmax(np.asarray(prediction)) == labelTest: #if the max of the query output is equal to the label then we increase count
			count += 1
		y = y + 1
	print("Accuracy: " + str(count / len(testValues)))
	print(datetime.now() - startTime)


if __name__ == "__main__":
    main()






