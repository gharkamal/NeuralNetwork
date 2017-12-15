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
		self.layer2weights= self.randomValues(784, 100)  #set the weights for the input -> hidden layer
		self.layer3weights= self.randomValues(100, 10) 	 #set the weights for the hidden -> output layer

		pass

	def train(self, inputs, l2w, l3w, label):
		#The two matrices have to be dot product compatible
		#Cols of matrix 1 should be equal to rows of matrix 2
		input2hidden = np.dot(l2w , inputs) #input layer and first set of weights 
		#print(str("input and weights dot: "))
		#print(input2hidden)
		hiddenVal = self.sigmoid(input2hidden) #value of hidden layer
		#print(str("Sigmoid layer 2"))
		#print(hiddenVal)

		hiddentoOut = np.dot(l3w, hiddenVal) #hidden layer and 2nd set of weights
		#print(str("hiddenOUt dot: "))
		#print(hiddentoOut)
		layer3Val = self.sigmoid(hiddentoOut) # output
		
		#print(layer3Val)
		#delta is the error (label and ouput value is used)
		delta = []
		for i, j in zip(label, layer3Val):
			delta.append(i - j)
		print(delta)
		pass

	def query():
		pass


	def getValues(self):
		filename ="temp/mnist_train.csv"
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
	nn = neuralNetwork("", "","","" )  #initialize neural network
	#Read vectors from file: first value is label and rest is vector in row 
	#image data converted to RGB value its a 28x28 matrix - then flattened 1D vector 784 pixels for each label 
	# input layer will ahve 784 nodes
	#seed forward , input weight forward passing 
	#weights = np.random.uniform(low=.1, high=.9, size=784)

	#generate random weihts tice once for in to hidden and hiden out
	values =  nn.getValues()
	#layer2weights= nn.randomValues(784, 100)
	#layer3weights= nn.randomValues(100, 10)
	i = 0
	while i < 1:
		normalVals = nn.normalizeVales(values[i][1:])
		#print(normalVals)
		label = values[i][0]
		labArr = nn.buildLabArr(label)
		#inputVal = values[0][1:]
		#print(values[0][1:])
		#print(fullWeights)
		nn.train(normalVals, nn.layer2weights, nn.layer3weights, labArr) 
		i = i + 1



if __name__ == "__main__":
    main()






