import sys
import numpy
import pandas as panda
import pandas as pd 

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
            print(predicted[j],actual[j])
            if(predicted[j][0] == actual[j][0]):
                accuracy = accuracy+1
            j = j+1
    else:
        return "Can't find accuracy of two differetn sized matrixs"
    return accuracy/maxAccuracyScore
def oneHotEncoding(owlData):
    #oneHotData = owlData[:,4]

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

    def negative_log_likelihood(self):
        # sigmoid_activation = sigmoid(numpy.dot(self.x, self.W) + self.b)
        sigmoid_activation = softmax(numpy.dot(self.x, self.W) + self.b)

        cross_entropy = - numpy.mean(
            numpy.sum(self.y * numpy.log(sigmoid_activation) +
            (1 - self.y) * numpy.log(1 - sigmoid_activation),
                      axis=1))

        return cross_entropy


    def predict(self, x):
        # return sigmoid(numpy.dot(x, self.W) + self.b)
        return softmax(numpy.dot(x, self.W) + self.b)


def test_lr(learning_rate=0.01, n_epochs=200):
    # training data
    #open the csv file
    owlData=numpy.array(panda.read_csv("owls15.csv"))
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
    print(targetData)
   #print(targetData) 
    #targetData  = oneHotEncoding(targetData)
    featureData = owlData[:,0:typeColumn]
    trainTargetData = targetData[trainLength:]
    trainFeatureData = featureData[trainLength:]
    testTargetData = targetData[:trainLength]
    testFeatureData = featureData[:trainLength]
    print(trainTargetData)
    #print(featureData)
    

    x = numpy.array(trainFeatureData,dtype=numpy.float32)
    '''[3,5,1.6,0.2],
                    [3.2,4.7,1.6,0.2],
                    [3.4,4.6,1.4,0.3],
                    [3.6,5,1.4,0.2],
[4.1,5.2,1.5,0.1],
[3,4.9,1.4,0.2],
[3.3,5.1,1.7,0.5],
[3.4,4.8,1.6,0.2],
[3.7,5.1,1.5,0.4],
[3.1,4.9,1.5,0.1],
[3.6,4.6,1,0.2],
[3.9,5.4,1.7,0.4],
[3,4.3,1.1,0.1],
[2.9,4.4,1.4,0.2],
[3.4,5,1.5,0.2],
[3.5,5.1,1.4,0.2],
[3.2,5,1.2,0.2],
[3.8,5.1,1.9,0.4],
[3,4.8,1.4,0.3],
[4.2,5.5,1.4,0.2],
[3.7,5.4,1.5,0.2],
[3,4.8,1.4,0.1],
[3.3,5,1.4,0.2],
[3.2,4.7,1.3,0.2],
[3.5,5.1,1.4,0.3],
[3.4,4.8,1.9,0.2],
[3.4,5.4,1.5,0.4],
[3.8,5.1,1.5,0.3],
[3,4.4,1.3,0.2],
[3.5,5,1.3,0.3],
[3.5,5,1.6,0.6],
[3.9,5.4,1.3,0.4],
[3.1,4.6,1.5,0.2],
[3.8,5.7,1.7,0.3],
[4,5.8,1.2,0.2],
[3.7,5.3,1.5,0.2],
[3.4,5,1.6,0.4],
[3.2,4.4,1.3,0.2],
[3.4,5.4,1.7,0.2],
[3.2,4.6,1.4,0.2],
[3.5,5.5,1.3,0.2],
[3.1,4.8,1.6,0.2],
[3.1,4.9,1.5,0.1],
[3.4,5.2,1.4,0.2],
[2.3,4.5,1.3,0.3],
[2.8,6.4,5.6,2.1],
[3.2,6.4,5.3,2.3],
[2.8,6.2,4.8,1.8],
[2.5,6.3,5,1.9],
[3,6.1,4.9,1.8],
[3.3,6.3,6,2.5],
[2.6,7.7,6.9,2.3],
[2.7,6.3,4.9,1.8],
[2.5,4.9,4.5,1.7],
[3,5.9,5.1,1.8],
[3.1,6.4,5.5,1.8],
[3.8,7.7,6.7,2.2],
[3.4,6.2,5.4,2.3],
[3.2,6.8,5.9,2.3],
[3.1,6.9,5.1,2.3],
[3,6.8,5.5,2.1],
[3.3,6.7,5.7,2.5],
[3.2,6.9,5.7,2.3],
[3,7.6,6.6,2.1],
[2.8,6.4,5.6,2.2],
[2.2,6,5,1.5],
[2.9,7.3,6.3,1.8],
[3,7.1,5.9,2.1],
[3,7.2,5.8,1.6],
[3,6,4.8,1.8],
[2.8,5.6,4.9,2],
[2.9,6.3,5.6,1.8],
[2.7,5.8,5.1,1.9],
[2.7,5.8,5.1,1.9],
[2.5,6.7,5.8,1.8],
[2.8,6.3,5.1,1.5],
[3.3,6.7,5.7,2.1],
[3.2,7.2,6,1.8],
[2.5,5.7,5,2],
[3.4,6.3,5.6,2.4],
[3.1,6.9,5.4,2.1],
[3,6.5,5.2,2],
[3,7.7,6.1,2.3],
[2.8,7.4,6.1,1.9],
[2.8,7.7,6.7,2],
[3.6,7.2,6.1,2.5],
[2.6,6.1,5.6,1.4],
[3,6.7,5.2,2.3],
[3,6.5,5.8,2.2],
[2.7,6.4,5.3,1.9],
[2.7,5.6,4.2,1.3],
[3,5.9,4.2,1.5],
[3,6.1,4.6,1.4],
[2.3,5.5,4,1.3],
[2.8,5.7,4.1,1.3],
[3,5.6,4.5,1.5],
[2.8,6.8,4.8,1.4],
[3.1,6.7,4.4,1.4],
[2.5,5.5,4,1.3],
[2.9,6.2,4.3,1.3],
[3.1,6.9,4.9,1.5],
[2.4,5.5,3.8,1.1],
[2.8,6.1,4.7,1.2],
[3,6.7,5,1.7],
[2.8,6.1,4,1.3],
[2.9,6.4,4.3,1.3],
[2.7,6,5.1,1.6],
[2.7,5.8,4.1,1],
[2.6,5.8,4,1.2],
[3,5.7,4.2,1.2],
[2.4,5.5,3.7,1],
[2.5,5.1,3,1.1],
[3.4,6,4.5,1.6],
[2.3,5,3.3,1],
[2.9,5.7,4.2,1.3],
[2.7,5.8,3.9,1.2],
[2.6,5.5,4.4,1.2],
[2,5,3.5,1],
[2.5,5.6,3.9,1.1],
[3,5.4,4.5,1.5],
[2.3,6.3,4.4,1.3],
[2.8,6.5,4.6,1.5],
[3,6.6,4.4,1.4],
[2.9,6,4.5,1.5],
[2.7,5.2,3.9,1.4],
[2.5,6.3,4.9,1.5],
[3.2,6.4,4.5,1.5],
[2.9,6.1,4.7,1.4],
[2.9,5.6,3.6,1.3],
[2.2,6,4,1],
[3.3,6.3,4.7,1.6],
[3.1,6.7,4.7,1.5],
[2.9,6.6,4.6,1.3],
[3,5.6,4.1,1.3],
[3.2,5.9,4.8,1.8]])'''
    y = numpy.array(trainTargetData,dtype=numpy.float32) 
    '''[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], 
        [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], 
        [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], 
        [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], 
        [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], 
        [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], 
        [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], 
        [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], 
        [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], 
        [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], 
        [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]])'''
    #x = scaleData(x)

    # construct LogisticRegression
    classifier = LogisticRegression(input=x, label=y, n_in=4, n_out=3)

    # train
    for epoch in range(n_epochs):
        classifier.train(lr=learning_rate)
        cost = classifier.negative_log_likelihood()
        #print('Training epoch %d, cost is ' % epoch, cost)
        learning_rate *= 0.95


    # test
    x = numpy.array(testFeatureData,dtype=numpy.float32)
    y = numpy.array(testTargetData,dtype=numpy.float32)
    '''[[3,5,1.6,0.2],
                    [3.2,4.7,1.6,0.2],
                    [3.4,4.6,1.4,0.3],
                    [3.6,5,1.4,0.2],
[4.1,5.2,1.5,0.1],
[3,4.9,1.4,0.2],
[3.3,5.1,1.7,0.5],
[3.4,4.8,1.6,0.2],
[3.7,5.1,1.5,0.4],
[3.1,4.9,1.5,0.1],
[3.6,4.6,1,0.2],
[3.9,5.4,1.7,0.4],
[3,4.3,1.1,0.1],
[2.9,4.4,1.4,0.2],
[3.4,5,1.5,0.2],
[3.5,5.1,1.4,0.2],
[3.2,5,1.2,0.2],
[3.8,5.1,1.9,0.4],
[3,4.8,1.4,0.3],
[4.2,5.5,1.4,0.2],
[3.7,5.4,1.5,0.2],
[3,4.8,1.4,0.1],
[3.3,5,1.4,0.2],
[3.2,4.7,1.3,0.2],
[3.5,5.1,1.4,0.3],
[3.4,4.8,1.9,0.2],
[3.4,5.4,1.5,0.4],
[3.8,5.1,1.5,0.3],
[3,4.4,1.3,0.2],
[3.5,5,1.3,0.3],
[3.5,5,1.6,0.6],
[3.9,5.4,1.3,0.4],
[3.1,4.6,1.5,0.2],
[3.8,5.7,1.7,0.3],
[4,5.8,1.2,0.2],
[3.7,5.3,1.5,0.2],
[3.4,5,1.6,0.4],
[3.2,4.4,1.3,0.2],
[3.4,5.4,1.7,0.2],
[3.2,4.6,1.4,0.2],
[3.5,5.5,1.3,0.2],
[3.1,4.8,1.6,0.2],
[3.1,4.9,1.5,0.1],
[3.4,5.2,1.4,0.2],
[2.3,4.5,1.3,0.3],
[2.8,6.4,5.6,2.1],
[3.2,6.4,5.3,2.3],
[2.8,6.2,4.8,1.8],
[2.5,6.3,5,1.9],
[3,6.1,4.9,1.8],
[3.3,6.3,6,2.5],
[2.6,7.7,6.9,2.3],
[2.7,6.3,4.9,1.8],
[2.5,4.9,4.5,1.7],
[3,5.9,5.1,1.8],
[3.1,6.4,5.5,1.8],
[3.8,7.7,6.7,2.2],
[3.4,6.2,5.4,2.3],
[3.2,6.8,5.9,2.3],
[3.1,6.9,5.1,2.3],
[3,6.8,5.5,2.1],
[3.3,6.7,5.7,2.5],
[3.2,6.9,5.7,2.3],
[3,7.6,6.6,2.1],
[2.8,6.4,5.6,2.2],
[2.2,6,5,1.5],
[2.9,7.3,6.3,1.8],
[3,7.1,5.9,2.1],
[3,7.2,5.8,1.6],
[3,6,4.8,1.8],
[2.8,5.6,4.9,2]])'''
   
    data = (classifier.predict(x))
    data = maxValToOne(data)
    print(numpy.c_[data, y])
    accuracy  = getAccuracy(data,y)
    print(accuracy)
    print(len(data))
if __name__ == "__main__":
    test_lr()