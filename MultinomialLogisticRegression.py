import numpy as np
import pandas as pd 
from tkinter.filedialog import askopenfilename
import math as Math
import pymsgbox as msBox
from collections import Counter

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

#find the unique owl types in the target column. Returns them in the order which they occur. 
def unique_char(owlData):

    uniqueTypes = []
    for value in owlData:
        if value not in uniqueTypes:
            uniqueTypes.append(value)
    return uniqueTypes
#Encode the owl types as vector representations using one-hot encoding. 
#Using this an owl of type "LongEaredOwl" will be represented as [1,0,0] We need to figure out
#how many unique types of owls are in the array.
def oneHotEncoding(owlData,types):
    character_to_int = dict((c, i) for i, c in enumerate(types))
    int_to_character = dict((i, c) for i, c in enumerate(types))
    # encode input data as integers 
    owlType_encoded = [character_to_int[char] for char in owlData]
    
    # apply one hot encoding 
    one_hot_encoding = list()
    for value in owlType_encoded:
        owlType = [0 for _ in range(len(types))]
        owlType[value] = 1
        one_hot_encoding.append(owlType)
    ##manual comparision of encoded values to ensure correctness
    #i=0
    #while (j <len(owlData)):
    #   print('{0}    vs      {1}\n'.format(one_hot_encoding[j],owlData[j]))
    #   j=j+1
    return one_hot_encoding

#after our prediction has been calculated using the softmax function we promote the highest 
#value in each row to  1 and the rest to a 0. This makes it easier to view the data and also
#to performt the getAccuracy function
def maxValToOne(data):
    #create array of 0's the same size as the current array
    correctData= np.zeros_like(data)
    correctData[np.arange(len(data)), data.argmax(1)] = 1
    return correctData

def returnOwlTypes(encodedData,types):
    predictedOwlTypes = list()
    print(types)
    j =  0
    while(j<len(encodedData)):
        i= 0
        while(i<len(types)):
            if(encodedData[j][i] == 1):
                #print('{0}  vs {1}\n'.format(encodedData[j], types[i]))
                predictedOwlTypes.append(types[i])
            i = i+1
        j = j+1
    #manual check to ensure the values returned are correct
    #j = 0
    #while(j<len(predictedOwlTypes)):
        #print('{0}    vs       {1}\n'.format(predictedOwlTypes[j],encodedData[j]))
        #j =j +1
    return predictedOwlTypes

#softmax function computes the predicted category by taking product of the predict function.  
def softmax(z):
    x = np.exp(z - np.max(z))
    if x.ndim == 1:
        return x / np.sum(x, axis=0)
    else:
        return x / np.array([np.sum(x, axis=1)]).T
    
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

def costFunc(features, target, weights,bias):
        
        softmax_cost = softmax(np.dot(features, weights) + bias)

        cost = - np.mean(np.sum(target* np.log(softmax_cost) +
            (1 - target) * np.log(1 - softmax_cost),axis=1))

        return cost

#function that our main calls when you execute the code. Loads the data,separates the data into train and test
def test_lr(alpha=0.001, num_Iterations=1000):

    #open window that allows user to select file to test.
    fileName = askopenfilename(title='Select a new file to test!')
    #select the file you wish to write to. We also clear the contents of the file first
    destination = askopenfilename(title='Select a file to write the results to!')
    open(destination, 'w').close()
    #open the csv file. We are assuming the user defined the headers for the data in the dataset
    owlData=np.array(pd.read_csv(fileName,header = None))
    #remove headers and separate the data
    headerData = owlData[0]
    owlData = owlData[1:]
    
    #find the length of one third of the data. The data set sill alway be split 2/3 training and 1/3 testing 
    #so it is safe to hardcode the valeu of 3
    trainLength = Math.floor(len(owlData)/3)
    #find the number of columns in the array, We are always going to assume that the last column holds the type 
    #i.e Barn owl, Snowy owl etc. Using this we will assume the rest of the columns to be feature data 
    typeColumn = (len(owlData[0])) - 1
    #find the unique categories in the type column. 
    catTypes = set(owlData[:, typeColumn])
    #find the number of categories. This is so we can construct our weight and bias matricies
    numCategories = len(catTypes)
    convertedToList = unique_char((owlData[:, typeColumn]))

    i=1
    combinedAccuracyScore=0
    #runs the code 10 times
    while(i<11):
    
        #randomise the data to ensure a fair sampling is taken for testing and training for each iteration
        np.random.shuffle(owlData)
       
        #extract the type data from the owl data. only the last column is taken out
        targetData = owlData[:,typeColumn]
        #find the number of ccategories in the dataset. easiest to do this before the one hot encoding starts
       
        #perform one hot encoding on the Target data. We place the returned data in np array for easier access and manipulation
        encodedTargetData = np.array(oneHotEncoding(targetData,catTypes))
    
        #extract the featureData from the owlData array. All columns bar the last are taken out as we are assuming that the last column will 
        #always be the type data
        featureData = owlData[:,0:typeColumn]

        #split the data set into train and test. 1/3rd test and 2/3rds train
        trainTargetData = encodedTargetData[trainLength:]
        trainFeatureData = featureData[trainLength:]
        testTargetData = encodedTargetData[:trainLength]
        testFeatureData = featureData[:trainLength]
    
        #create our feature array and target arrys for testing 
        featureDataArr = np.array(trainFeatureData,dtype=np.float32)
        targetDataArr = np.array(trainTargetData,dtype=np.float32) 

        #find number of features in the dataset
        numFeatures = len(featureDataArr.T)
        #find the number of ccategories in the dataset
   
        #initialize the weight and bias matricies. Both start with all 0's
        weightMatrix = np.zeros((numFeatures, numCategories))  
        biasMatrix = np.zeros(numCategories) 
        f = open(destination, 'a')
        # train the data set. Iterate number of times as defined in the function definition(1000 time with a learning rate of 0.001)
        #settled on these values after it returned the best values after multiple runs
        for num_Iters in range(num_Iterations):
            weights,bias = train(featureDataArr,targetDataArr,weightMatrix,biasMatrix,alpha)
            cost  = costFunc(featureDataArr,targetDataArr,weightMatrix,biasMatrix)
            if(num_Iters % 500 == 0):
                    f.write('Iteration =  {0} Cost = {1}\n'.format(num_Iters, cost) )    
            #update the weightMatrix every time it is return after each iterations
            weightMatrix= weights
            biasMatrix =bias


        # now we create our test arrays
        x1 = np.array(testFeatureData,dtype=np.float32)
        y1 = np.array(testTargetData,dtype=np.float32)
        #run our prediction based on the weight and bias matricies calculated in the train step above
        predictedType = predict(x1,weightMatrix,biasMatrix)
        #turn the predictions back into ones and zero's for easier comparision
        predictedType = maxValToOne(predictedType)
        #print(np.c_[data, y1])
        #compute the accuracy score. Test the prediced results verus the actual results
        accuracy  = getAccuracy(predictedType,y1)
        #keep track of the accuracy scored
        predictedOwlType = returnOwlTypes(predictedType,convertedToList)
        targetType = returnOwlTypes(y1,convertedToList)
        combinedAccuracyScore += accuracy
        f = open(destination, 'a')
        f.write("Accruacy : {0}% for iteration {1} \n".format(accuracy , i))
        f.write('Predicted scores\t       Actual Scores\n')
        j = 0
        while (j <len(predictedType)):
            f.write('{0:<15}    vs      {1:<15}\n'.format(predictedOwlType[j],targetType[j]))
            j=j+1
        i = i +1
        f.write('\n----------------------------------------------------------------------\n')     
    #compute the average score after 10 iterations with data being randomised each time. Print it to the bottom of the text file
    averageAccuracyScore = combinedAccuracyScore/(i-1) 
    f.write('Average Accuracy after {0} random iterations = {1}%'.format((i-1), averageAccuracyScore)) 
if __name__ == "__main__":
    test_lr()