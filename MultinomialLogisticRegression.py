import sys
import numpy as np
import pandas as panda
import pandas as pd 
from tkinter.filedialog import askopenfilename
import math as Math

np.seterr(all='ignore')
 
def sigmoid(x):
    return 1. / (1 + np.exp(-x))
def getAccuracy(predicted, actual):
    accuracy = 0
    maxAccuracyScore = len(predicted)
    if(len(predicted) == len(actual)):
        j=0
        while(j<len(predicted)):
            #print(predicted[j],actual[j])
            if(predicted[j][0] == actual[j][0]):
                accuracy = accuracy+1
            j = j+1
    else:
        return "Can't find accuracy of two differetn sized matrixs"
    return accuracy/maxAccuracyScore
def oneHotEncoding(owlData,):
    #oneHotData = owlData[:,4]
    types = np.unique(owlData)
    alphabet = "LongEaredOwl","SnowyOwl","BarnOwl"

    char_to_int = dict((c, i) for i, c in enumerate(alphabet))
    int_to_char = dict((i, c) for i, c in enumerate(alphabet))
    # integer encode input data
    integer_encoded = [char_to_int[char] for char in owlData]
    #print(integer_encoded)

    # one hot encode
    onehot_encoded = list()
    for value in integer_encoded:
        letter = [0 for _ in range(len(alphabet))]
        letter[value] = 1
        onehot_encoded.append(letter)
    #print(onehot_encoded)
    # invert encoding
    #inverted = int_to_char[np.argmax(onehot_encoded[0])]
    #print(inverted)
    return onehot_encoded

def maxValToOne(data):
    #create array of 0's the same size as the current array
    correctData= np.zeros_like(data)
    correctData[np.arange(len(data)), data.argmax(1)] = 1
    return correctData

def scaleData(data):
    i=0
    while(i<3):
        #find max and min value of array
        maxVal = np.amax(data[:,i])
        minVal = np.amin(data[:,i])
        j=0
        while(j<len(data)):
            #compute new normailsed value and round to 4 decimals places. still ensures accuracy 
            newVal = round((data[j][i] - minVal)/(maxVal - minVal),4)
            data[j][i] = newVal
            j = j+1
        i=i+1
    return data
#normalise the data use mean and standard deviation
def normaliseData(data): 
    i=0
    while(i<4):
        #find max and min value of array
        maxVal = np.amax(data[:,i])
        minVal = np.amin(data[:,i])
        j=0
        while(j<len(data)):
            #compute new normailsed value and round to 4 decimals places. still ensures accuracy 
            newVal = round((data[j][i] - minVal)/(maxVal - minVal),4)
            data[j][i] = newVal
            j = j+1
        i=i+1
    return data
def softmax(z):
    e = np.exp(z - np.max(z))  # prevent overflow
    if e.ndim == 1:
        return e / np.sum(e, axis=0)
    else:  
        return e / np.array([np.sum(e, axis=1)]).T  # ndim = 2



def train(features, target,weights, bias, alpha,L2_reg=0.00):
    # p_y_given_x = sigmoid(np.dot(self.x, self.W) + self.b)
    p_y_given_x = softmax(np.dot(features, weights) + bias)
    d_y = target - p_y_given_x
        
    weights += alpha * np.dot(features.T, d_y) - alpha * L2_reg * weights
    bias += alpha * np.mean(d_y, axis=0)
    return weights,bias
        # cost = self.negative_log_likelihood()
        # return cost
def predict(features, weights, bias):
 # return sigmoid(np.dot(x, self.W) + self.b)
    return softmax(np.dot(features, weights) + bias)


def test_lr(learning_rate=0.01, n_epochs=200):
    # training data
    #open window that allows user to select file to test. 
    #fileName = askopenfilename()
    #open the csv file. We are assuming the user defined the headers for the data in the dataset
    owlData=np.array(panda.read_csv("owls15.csv",header = None))
    #remove headers 
    headerData = owlData[0]
    owlData = owlData[1:]
    np.random.shuffle(owlData)
    #find the number of columns in the array, We are always going to assume that the last column holds the type 
    #i.e Barn owl, Snowy owl etc. Using this we will assume the rest of the columns to be feature data 
    typeColumn = (len(owlData[0])) - 1
    #feature columns is 0 to 1 less than type column
    featureColumns  = typeColumn -1

    #find the length of one third of the data 
    trainLength = Math.floor(len(owlData)/3)
    targetData = owlData[:,typeColumn]

    targetData = np.array(oneHotEncoding(targetData))
    #print(targetData)
    #print(targetData) 
    #targetData  = oneHotEncoding(targetData)
    featureData = owlData[:,0:typeColumn]
    trainTargetData = targetData[trainLength:]
    trainFeatureData = featureData[trainLength:]
    testTargetData = targetData[:trainLength]
    testFeatureData = featureData[:trainLength]
    #print(trainTargetData)
    #print(featureData)
    

    x = np.array(trainFeatureData,dtype=np.float32)
   
    y = np.array(trainTargetData,dtype=np.float32) 
    #find number of features in the dataset
    numFeatures = len(x.T)
   #find the number of ccategories in the dataset
    numCategories = len(y.T)
   
    weightMatrix = np.zeros((numFeatures, numCategories))  
    biasMatrix = np.zeros(numCategories) 

    # train
    for epoch in range(n_epochs):
        weights,bias = train(x,y,weightMatrix,biasMatrix,alpha=learning_rate)
        if (epoch % 100 ==0):
            print('Training epoch %d, cost is ' % epoch)
        learning_rate *= 0.95
        weightMatrix= weights
        biasMatrix =bias


    # test
    x1 = np.array(testFeatureData,dtype=np.float32)
    y1 = np.array(testTargetData,dtype=np.float32)
   
    data = predict(x1,weightMatrix,biasMatrix)
    data = maxValToOne(data)
    print(np.c_[data, y1])
    accuracy  = getAccuracy(data,y1)
    print(accuracy)
    #print(len(data))
if __name__ == "__main__":
    test_lr()