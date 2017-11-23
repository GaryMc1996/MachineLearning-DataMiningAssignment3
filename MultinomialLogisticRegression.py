import sys
import numpy
import pandas as panda
import pandas as pd 
from tkinter.filedialog import askopenfilename
import math as Math

numpy.seterr(all='ignore')
 
def sigmoid(x):
    return 1. / (1 + numpy.exp(-x))
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
    types = numpy.unique(owlData)
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
    correctData= numpy.zeros_like(data)
    correctData[numpy.arange(len(data)), data.argmax(1)] = 1
    return correctData

def scaleData(data):
    i=0
    while(i<3):
        #find max and min value of array
        maxVal = numpy.amax(data[:,i])
        minVal = numpy.amin(data[:,i])
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
        maxVal = numpy.amax(data[:,i])
        minVal = numpy.amin(data[:,i])
        j=0
        while(j<len(data)):
            #compute new normailsed value and round to 4 decimals places. still ensures accuracy 
            newVal = round((data[j][i] - minVal)/(maxVal - minVal),4)
            data[j][i] = newVal
            j = j+1
        i=i+1
    return data
def softmax(x):
    e = numpy.exp(x - numpy.max(x))  # prevent overflow
    if e.ndim == 1:
        return e / numpy.sum(e, axis=0)
    else:  
        return e / numpy.array([numpy.sum(e, axis=1)]).T  # ndim = 2


class LogisticRegression(object):
    def __init__(self, input, label, n_in, n_out):
        self.x = input
        self.y = label
        self.W = numpy.zeros((n_in, n_out))  # initialize W 0
        self.b = numpy.zeros(n_out)          # initialize bias 0

        # self.params = [self.W, self.b]

    def train(self, lr=0.1, input=None, L2_reg=0.00):
        if input is not None:
            self.x = input

        # p_y_given_x = sigmoid(numpy.dot(self.x, self.W) + self.b)
        p_y_given_x = softmax(numpy.dot(self.x, self.W) + self.b)
        d_y = self.y - p_y_given_x
        
        self.W += lr * numpy.dot(self.x.T, d_y) - lr * L2_reg * self.W
        self.b += lr * numpy.mean(d_y, axis=0)

        # cost = self.negative_log_likelihood()
        # return cost




    def predict(self, x):
        # return sigmoid(numpy.dot(x, self.W) + self.b)
        return softmax(numpy.dot(x, self.W) + self.b)


def test_lr(learning_rate=0.01, n_epochs=200):
    # training data
    #open window that allows user to select file to test. 
    #fileName = askopenfilename()
    #open the csv file. We are assuming the user defined the headers for the data in the dataset
    owlData=numpy.array(panda.read_csv("owls15.csv",header = None))
    #remove headers 
    headerData = owlData[0]
    owlData = owlData[1:]
    numpy.random.shuffle(owlData)
    #find the number of columns in the array, We are always going to assume that the last column holds the type 
    #i.e Barn owl, Snowy owl etc. Using this we will assume the rest of the columns to be feature data 
    typeColumn = (len(owlData[0])) - 1
    #feature columns is 0 to 1 less than type column
    featureColumns  = typeColumn -1

    #find the length of one third of the data 
    trainLength = Math.floor(len(owlData)/3)
    targetData = owlData[:,typeColumn]

    targetData = numpy.array(oneHotEncoding(targetData))
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
    

    x = numpy.array(trainFeatureData,dtype=numpy.float32)
   
    y = numpy.array(trainTargetData,dtype=numpy.float32) 

    # construct LogisticRegression
    classifier = LogisticRegression(input=x, label=y, n_in=4, n_out=3)

    # train
    for epoch in range(n_epochs):
        classifier.train(lr=learning_rate)
        
        if (epoch % 100 ==0):
            print('Training epoch %d, cost is ' % epoch)
        learning_rate *= 0.95


    # test
    x1 = numpy.array(testFeatureData,dtype=numpy.float32)
    y1 = numpy.array(testTargetData,dtype=numpy.float32)
   
    data = (classifier.predict(x1))
    data = maxValToOne(data)
    #print(numpy.c_[data, y])
    accuracy  = getAccuracy(data,y1)
    print(accuracy)
    #print(len(data))
if __name__ == "__main__":
    test_lr()