import sys
import numpy as np
import pandas as panda
import pandas as pd 
from tkinter.filedialog import askopenfilename
import math as Math

#find accuracy of the algorithm. Essentially check every position of the prediction array agaisnt its
#corresponding position in the array with the actual value. Every time they match we give it a scor eof +1
#at the end them we can get a percentile score by dividing by the max score. Which would be equal to the 
#length of the arrays. We also include a check to ensure that the arrays are the same length. We round also 
#to return a nice clean score in the event of a fraction
def getAccuracy(predicted, actual):
    accuracy = 0
    maxAccuracyScore = len(predicted)
    if(len(predicted) == len(actual)):
        j=0
        while(j<len(predicted)):
            if(predicted[j][0] == actual[j][0]):
                accuracy = accuracy+1
            j = j+1
    else:
        return "Can't find accuracy of two differenn sized matrixs"
    return round(accuracy/maxAccuracyScore*100,4)

#Encode the owl types as vector representations using one-hot encoding. 
#Using this an owl of type "LongEaredOwl" will be represented as [1,0,0] We need to figure out
#how many unique types of owls are in the array.
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
    
    return onehot_encoded

#after our prediction has been calculated using the softmax function we promote the highest 
#value in each row to  1 and the rest to a 0. This makes it easier to view the data and also
#to performt the getAccuracy function
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
#softmax function computes the predicted category by taking product of the predict function.  
def softmax(z):
    e = np.exp(z - np.max(z))  # prevent overflow
    if e.ndim == 1:
        return e / np.sum(e, axis=0)
    else:  
        return e / np.array([np.sum(e, axis=1)]).T  # ndim = 2

#Train your Dataset. Here we train our bias and weihght matrix. Both initially start off as 0 matricies
def train(features, target,weights, bias, alpha):

    #get the current prediction based on our current bias and weight matricies
    currentPrediction = softmax(np.dot(features, weights) + bias)
    #check to see how far away our prediction was from the actual values
    error = target - currentPrediction
    #update the weight and bias matricies    
    weights += alpha * np.dot(features.T, error) - alpha * weights
    bias += alpha * np.mean(error, axis=0)
    #return the weights and bias matricies now that the optimium score has been computed
    return weights,bias

#predict function that calculates the net score based on the dot product of the features and weight matricies. 
#The constant bias is then added to this. This value is then passed to the softmax function
def predict(features, weights, bias):
    return softmax(np.dot(features, weights) + bias)

#function that our main calls when you execute the code. Loads the data,separates the data into train and test
def test_lr(alpha=0.001, num_Iterations=1000):
    # training data
    #open window that allows user to select file to test. 
    #fileName = askopenfilename()
    #open the csv file. We are assuming the user defined the headers for the data in the dataset
    owlData=np.array(panda.read_csv("owls15.csv",header = None))
    #remove headers
    i=1
    combinedAccuracyScore=0
    while(i<11):
        headerData = owlData[0]
        owlData = owlData[1:]
        #randomise the data to ensure a fair sampling is taken for testing and training
        np.random.shuffle(owlData)
        #find the number of columns in the array, We are always going to assume that the last column holds the type 
        #i.e Barn owl, Snowy owl etc. Using this we will assume the rest of the columns to be feature data 
        typeColumn = (len(owlData[0])) - 1
        #extract the type data from the owl data. only the last column is taken out
        targetData = owlData[:,typeColumn]

        #find the length of one third of the data. The data set sill alway be split 2/3 training and 1/3 testing 
        #so it is safe to hardcode the valeu of 3
        trainLength = Math.floor(len(owlData)/3)

        #perform one hot encoding on the Target data. We place the returned data in numpy array for easier access and manipulation
        targetData = np.array(oneHotEncoding(targetData))
    
        #extract the featureData from the owlData array. All columns bar the last are taken out as we are assuming that the last column will 
        #always be the type data
        featureData = owlData[:,0:typeColumn]

        #split the data set into train and test. 1/3rd test and 2/3rds train
        trainTargetData = targetData[trainLength:]
        trainFeatureData = featureData[trainLength:]
        testTargetData = targetData[:trainLength]
        testFeatureData = featureData[:trainLength]
    
        #create our feature array and target arrys for testing 
        featureData = np.array(trainFeatureData,dtype=np.float32)
        targetData = np.array(trainTargetData,dtype=np.float32) 

        #find number of features in the dataset
        numFeatures = len(featureData.T)
        #find the number of ccategories in the dataset
        numCategories = len(targetData.T)
   
        #initialize the weight and bias matricies. Both start with all 0's
        weightMatrix = np.zeros((numFeatures, numCategories))  
        biasMatrix = np.zeros(numCategories) 

        # train the data set. Iterate number of times as defined in the function definition(1000 time with a learning rate of 0.001)
        #settled on these values after it returned the best values after multiple runs
        for num_Iters in range(num_Iterations):
            weights,bias = train(featureData,targetData,weightMatrix,biasMatrix,alpha)
            #update the weightMatrix every time it is return after each iterations
            weightMatrix= weights
            biasMatrix =bias


        # now we create our test arrays
        x1 = np.array(testFeatureData,dtype=np.float32)
        y1 = np.array(testTargetData,dtype=np.float32)
        #run our prediction based on the weight and bias matricies calculated in the train step above
        data = predict(x1,weightMatrix,biasMatrix)
        #turn the predictions back into ones and zero's for easier comparision
        data = maxValToOne(data)
        #compute the accuracy score. Test the prediced results verus the actual results
        accuracy  = getAccuracy(data,y1)
        print(accuracy,'% Accuracy for iteration',i)
        i = i +1
        combinedAccuracyScore += accuracy
    ##compute the average score after 10 iterations with data being randomised each time
    averageAccuracyScore = combinedAccuracyScore/(i-1) 
    print('Average Accuracy after %d random iterations = '% (i-1), averageAccuracyScore) 
if __name__ == "__main__":
    test_lr()