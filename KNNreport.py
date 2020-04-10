# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 21:50:24 2018

@author: Ryutaro Takanami
"""
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import time

#Load Pulsar
dataRaw = [];
DataFile = open ("winequality-red.csv", "r")

count=0
while True:
    theline = DataFile.readline()
    if count ==0:
        count += 1
        continue
    if len(theline) == 0:
        break
    readData = theline.split(";")
    for pos in range(len(readData)):
        readData[pos] = float(readData[pos]);
    dataRaw.append(readData)
    
DataFile.close()

wineData = np.array(dataRaw)
df = pd.read_csv("winequality-red.csv", sep=";")
df2 =  pd.read_csv("winequality-white.csv", sep=";")
df = pd.concat([df, df2], ignore_index = True)

df.loc[~(df['quality'] == 6), 'quality'] = 0
df.loc[df['quality'] == 6, 'quality'] = 1

wineLabel = wineData[:,11]
wine = wineData[:,0:11]

print(df['quality'].value_counts())


##################################################################################################
#KNN

class KNN:

    def predict(test, train, trainLabel, k):
        preLabel_list = []
        for test_line in test:
            #subtract the test data from all train data
            diff = train - test_line
            #square the differences
            sqr = diff ** 2
            #sum up all squared value
            sumD = sqr.sum(axis=1)
            #root summed data to compute Euclidean distance
            dist = sumD ** 0.5
            #sort distance data
            index = dist.argsort()
            #use K data to predict the label of the test data
            k_label = trainLabel[index[0:k]]
            #count the frequency of the label
            count = np.bincount(k_label.astype(int))
            #predict the label as mode (if some labels have same frequency, predct as the smaller label)
            preLabel = np.argmax(count)
            #add the predicted label to the list
            preLabel_list.append(preLabel)
            
        return preLabel_list
    

##################################################################################################
#Kfold
        




#generate data to implement 5 fold cross validation
class Kfold:
        
    def generateTrainTest(df, sampleNum):
        #select the number of data
        df_randomOrder = df.sample(n=sampleNum)
        
        
        
        #Divide test and train data
        n = int(df_randomOrder.shape[0]/5)
        test0 = df_randomOrder.iloc[0:n]
        test1 = df_randomOrder.iloc[n:2*n]
        test2 = df_randomOrder.iloc[2*n:3*n]
        test3 = df_randomOrder.iloc[3*n:4*n]
        test4 = df_randomOrder.iloc[4*n:]
        
        #Reset index for later part (for identifying the number of row)
        test0 = test0.reset_index(drop=True)
        test1 = test1.reset_index(drop=True)
        test2 = test2.reset_index(drop=True)
        test3 = test3.reset_index(drop=True)
        test4 = test4.reset_index(drop=True)
        
        #generate train data by combining test data
        train0 = pd.concat([test1, test2, test3, test4], ignore_index = True)
        train1 = pd.concat([test0, test2, test3, test4], ignore_index = True)
        train2 = pd.concat([test0, test1, test3, test4], ignore_index = True)
        train3 = pd.concat([test0, test1, test2, test4], ignore_index = True)
        train4 = pd.concat([test0, test1, test2, test3], ignore_index = True)
    
        return test0, test1, test2, test3, test4, train0, train1, train2, train3, train4
    
    
    def generateAll(df, sampleNum):
        #df = pd.DataFrame(wineData)
        test0, test1, test2, test3, test4, train0, train1, train2, train3, train4 = Kfold.generateTrainTest(df, sampleNum)
        
        
        
        
        train0 = train0.values
        train1 = train1.values
        train2 = train2.values
        train3 = train3.values
        train4 = train4.values
        
        test0 = test0.values
        test1 = test1.values
        test2 = test2.values
        test3 = test3.values
        test4 = test4.values
        
        train0Label = train0[:,11]
        train0 = train0[:,0:11]
        train1Label = train1[:,11]
        train1 = train1[:,0:11]
        train2Label = train2[:,11]
        train2 = train2[:,0:11]
        train3Label = train3[:,11]
        train3 = train3[:,0:11]
        train4Label = train4[:,11]
        train4 = train4[:,0:11]
        
        test0Label = test0[:,11]
        test0 = test0[:,0:11]
        test1Label = test1[:,11]
        test1 = test1[:,0:11]
        test2Label = test2[:,11]
        test2 = test2[:,0:11]
        test3Label = test3[:,11]
        test3 = test3[:,0:11]
        test4Label = test4[:,11]
        test4 = test4[:,0:11]
        
        return test0, test1, test2, test3, test4, train0, train1, train2, train3, train4, train0Label, train1Label, train2Label, train3Label, train4Label, test0Label, test1Label, test2Label, test3Label, test4Label
##################################################################################################################
#prediction with K-fold

#Library
from sklearn.neighbors import KNeighborsClassifier


def predict_lib(kNum):
    lib_accuracyRate_list = []
    lib_precision_list = []
    lib_recall_list = []

    
    for k in kNum:
        accuracyRate = 0
        precision = 0
        recall = 0
        print(k)
        #Cross validation
        #train the data, predict the label and calcurate the accuracy rate by each fold
        #define the number of nearest neighbours
        #knc = KNeighborsClassifier(n_neighbors=k, weights= "uniform")
        knc = KNeighborsClassifier(n_neighbors=k, weights= "distance")
        #train the data
        knc.fit(train0, train0Label)
        #predict the lavel
        pred0 = knc.predict(test0)
        #calcurate the accuracy rate, precision and recall
        accurate = 0
        truePositive = 0
        for n in range(len(pred0)):
            if(int(test0Label[n]) == pred0[n]):
                accurate += 1
            if((int(test0Label[n]) == 1) & (pred0[n] ==1)):
                truePositive += 1
        accuracyRate += (accurate/len(pred0))
        precision += truePositive/np.count_nonzero(pred0==1)
        recall += truePositive/np.count_nonzero(test0Label==1)
        
        #knc = KNeighborsClassifier(n_neighbors=k, weights= "uniform")
        knc = KNeighborsClassifier(n_neighbors=k, weights= "distance")
        knc.fit(train1, train1Label)
        pred1 = knc.predict(test1)
        #calcurate the accuracy rate, precision and recall
        accurate = 0
        truePositive = 0
        for n in range(len(pred1)):
            if(int(test1Label[n]) == pred1[n]):
                accurate += 1
            if((int(test1Label[n]) == 1) & (pred1[n] ==1)):
                truePositive += 1
        accuracyRate += (accurate/len(pred1))
        precision += truePositive/np.count_nonzero(pred1==1)
        recall += truePositive/np.count_nonzero(test1Label==1)
        
        #knc = KNeighborsClassifier(n_neighbors=k, weights= "uniform")
        knc = KNeighborsClassifier(n_neighbors=k, weights= "distance")
        knc.fit(train2, train2Label)
        pred2 = knc.predict(test2)
        #calcurate the accuracy rate, precision and recall
        accurate = 0
        truePositive = 0
        for n in range(len(pred2)):
            if(int(test2Label[n]) == pred2[n]):
                accurate += 1
            if((int(test2Label[n]) == 1) & (pred2[n] ==1)):
                truePositive += 1
        accuracyRate += (accurate/len(pred2))
        precision += truePositive/np.count_nonzero(pred2==1)
        recall += truePositive/np.count_nonzero(test2Label==1)
        
        #knc = KNeighborsClassifier(n_neighbors=k, weights= "uniform")
        knc = KNeighborsClassifier(n_neighbors=k, weights= "distance")
        knc.fit(train3, train3Label)
        pred3 = knc.predict(test3)
        #calcurate the accuracy rate, precision and recall
        accurate = 0
        truePositive = 0
        for n in range(len(pred3)):
            if(int(test3Label[n]) == pred3[n]):
                accurate += 1
            if((int(test3Label[n]) == 1) & (pred3[n] ==1)):
                truePositive += 1
        accuracyRate += (accurate/len(pred3))
        precision += truePositive/np.count_nonzero(pred3==1)
        recall += truePositive/np.count_nonzero(test3Label==1)
        
        
        #knc = KNeighborsClassifier(n_neighbors=k, weights= "uniform")
        knc = KNeighborsClassifier(n_neighbors=k, weights= "distance")
        knc.fit(train4, train4Label)
        pred4 = knc.predict(test4)
        #calcurate the accuracy rate, precision and recall
        accurate = 0
        truePositive = 0
        for n in range(len(pred4)):
            if(int(test4Label[n]) == pred4[n]):
                accurate += 1
            if((int(test4Label[n]) == 1) & (pred4[n] ==1)):
                truePositive += 1
        accuracyRate += (accurate/len(pred4))
        precision += truePositive/np.count_nonzero(pred4==1)
        recall += truePositive/np.count_nonzero(test4Label==1)
        
        
        #append the accuracy rate. precision and recall by each the number of neighbors
        lib_accuracyRate_list.append(accuracyRate/5)
        lib_precision_list.append(precision/5)
        lib_recall_list.append(recall/5)
    return lib_accuracyRate_list, lib_precision_list, lib_recall_list




#Scratch

def predict_scr(kNum):
    scr_accuracyRate_list = []
    scr_precision_list = []
    scr_recall_list = []
    
    for k in kNum:
        accuracyRate = 0
        precision = 0
        recall = 0
        print(k)
        
        #Cross validation
        #train the data, predict the label and calcurate the accuracy rate by each fold
        
        #predict the lavel
        pred0 = KNN.predict(test0, train0, train0Label, k)
        #calcurate the accuracy rate, precision and recall
        accurate = 0
        truePositive = 0
        for n in range(len(pred0)):
            if(int(test0Label[n]) == pred0[n]):
                accurate += 1
            if((int(test0Label[n]) == 1) & (pred0[n] ==1)):
                truePositive += 1
        accuracyRate += (accurate/len(pred0))
        precision += truePositive/pred0.count(1)
        recall += truePositive/np.count_nonzero(test0Label==1)
        
        pred1 = KNN.predict(test1, train1, train1Label, k)
        #calcurate the accuracy rate, precision and recall
        accurate = 0
        truePositive = 0
        for n in range(len(pred1)):
            if(int(test1Label[n]) == pred1[n]):
                accurate += 1
            if((int(test1Label[n]) == 1) & (pred1[n] ==1)):
                truePositive += 1
        accuracyRate += (accurate/len(pred1))
        precision += truePositive/pred1.count(1)
        recall += truePositive/np.count_nonzero(test1Label==1)
        
        pred2 = KNN.predict(test2, train2, train2Label, k)
        #calcurate the accuracy rate, precision and recall
        accurate = 0
        truePositive = 0
        for n in range(len(pred2)):
            if(int(test2Label[n]) == pred2[n]):
                accurate += 1
            if((int(test2Label[n]) == 1) & (pred2[n] ==1)):
                truePositive += 1
        accuracyRate += (accurate/len(pred2))
        precision += truePositive/pred2.count(1)
        recall += truePositive/np.count_nonzero(test2Label==1)
        
        pred3 = KNN.predict(test3, train3, train3Label, k)
        #calcurate the accuracy rate, precision and recall
        accurate = 0
        truePositive = 0
        for n in range(len(pred3)):
            if(int(test3Label[n]) == pred3[n]):
                accurate += 1
            if((int(test3Label[n]) == 1) & (pred3[n] ==1)):
                truePositive += 1
        accuracyRate += (accurate/len(pred3))
        precision += truePositive/pred3.count(1)
        recall += truePositive/np.count_nonzero(test3Label==1)
        
        pred4 = KNN.predict(test4, train4, train4Label, k)
        #calcurate the accuracy rate, precision and recall
        accurate = 0
        truePositive = 0
        for n in range(len(pred4)):
            if(int(test4Label[n]) == pred4[n]):
                accurate += 1
            if((int(test4Label[n]) == 1) & (pred4[n] ==1)):
                truePositive += 1
        accuracyRate += (accurate/len(pred4))
        precision += truePositive/pred4.count(1)
        recall += truePositive/np.count_nonzero(test4Label==1)
        
        #append the accuracy rate. precision and recall by each the number of neighbors
        scr_accuracyRate_list.append(accuracyRate/5)
        scr_precision_list.append(precision/5)
        scr_recall_list.append(recall/5)
    return scr_accuracyRate_list, scr_precision_list, scr_recall_list


################################################################################################


#Generate Data
sampleNum = df.shape[0]
test0, test1, test2, test3, test4, train0, train1, train2, train3, train4,\
train0Label, train1Label, train2Label, train3Label, train4Label,\
test0Label, test1Label, test2Label, test3Label, test4Label = Kfold.generateAll(df, sampleNum)


#############################Examine accuracy, precision and recall###################################

#find the best parameter k by changing k from 1 to 500
kNum = list(range(1, 501, 1))
lib_accuracyRate_list, lib_precision_list, lib_recall_list = predict_lib(kNum)
#save the accuracy rate list as a csv file
lib_accuracyRate = pd.Series(lib_accuracyRate_list)
lib_precision_list = pd.Series(lib_precision_list)
lib_recall_list = pd.Series(lib_recall_list)
lib_accuracyRate.to_csv("SCC461\lib_accuracyRate.csv", index=False)
lib_precision_list.to_csv("SCC461\lib_precision_list.csv", index=False)
lib_recall_list.to_csv("SCC461\lib_recall_list.csv", index=False)

#find the best parameter k by changing k from 1 to 500
kNum = list(range(1, 501, 1))
scr_accuracyRate_list, scr_precision_list, scr_recall_list = predict_scr(kNum)
#save the accuracy rate list as a csv file
scr_accuracyRate = pd.Series(scr_accuracyRate_list)
scr_precision_list = pd.Series(scr_precision_list)
scr_recall_list = pd.Series(scr_recall_list)
scr_accuracyRate.to_csv("SCC461\scr_accuracyRate.csv", index=False)
scr_precision_list.to_csv("SCC461\scr_precision_list.csv", index=False)
scr_recall_list.to_csv("SCC461\scr_recall_list.csv", index=False)



############################Examine changes of computing time####################################

#library
#record the time of computing with 100 trials
trial = 100
lib_time_list = []
kNum = list(range(1, 2, 1))
sampleNum_list = list(range(1000, df.shape[0], 500))

lib_time = pd.DataFrame({
        "Sample Number":[],
        "Trial":[],
        "Time":[]
        })


for sampleNum in sampleNum_list:
    test0, test1, test2, test3, test4, train0, train1, train2, train3, train4,\
    train0Label, train1Label, train2Label, train3Label, train4Label,\
    test0Label, test1Label, test2Label, test3Label, test4Label = Kfold.generateAll(df, sampleNum)
    for tryNum in range(trial):
    

        start = time.process_time()
        aaa = predict_lib(kNum)
        comTime = time.process_time() - start
        s = pd.DataFrame({
        "Sample Number":[sampleNum],
        "Trial":[tryNum+1],
        "Time":[comTime]
        })
        lib_time = lib_time.append(s, ignore_index=True)
lib_time.to_csv("SCC461\lib_time.csv", index=False)



#scratch
#record the time of computing with 100 trials
trial = 100
scr_time_list = []
kNum = list(range(1, 2, 1))
sampleNum_list = list(range(1000, df.shape[0], 500))

scr_time = pd.DataFrame({
        "Sample Number":[],
        "Trial":[],
        "Time":[]
        })
    
for sampleNum in sampleNum_list:
    test0, test1, test2, test3, test4, train0, train1, train2, train3, train4,\
    train0Label, train1Label, train2Label, train3Label, train4Label,\
    test0Label, test1Label, test2Label, test3Label, test4Label = Kfold.generateAll(df, sampleNum)
    for tryNum in range(trial):
    

        start = time.process_time()
        aaa = predict_scr(kNum)
        comTime = time.process_time() - start
        s = pd.DataFrame({
        "Sample Number":[sampleNum],
        "Trial":[tryNum+1],
        "Time":[comTime]
        })
        scr_time = scr_time.append(s, ignore_index=True)
scr_time.to_csv("SCC461\scr_time.csv", index=False)

"""
#Delete?
for i in range(trial):
    
    kNum = list(range(1, 2, 1))
    start = time.process_time()
    aaa = predict_lib(kNum)
    lib_time_list.append(time.process_time() - start)
lib_time_list = pd.Series(lib_time_list)
lib_time_list.to_csv("SCC461\lib_time_list.csv", index=False)
"""


"""
plt.plot(kNum, accuracyRate_list)
plt.xlabel("The number of nearest neighbors")
plt.ylabel("The accuracy rate")
"""










"""
plt.plot(kNum, accuracyRate_list)
plt.xlabel("The number of nearest neighbors")
plt.ylabel("The accuracy rate")
"""







"""


"""


"""
#Delete?
#record the time of computing with 100 trials
trial = 100
scr_time_list = []

for i in range(trial):
    
    kNum = list(range(1, 2, 1))
    start = time.process_time()
    bbb = predict_scr(kNum)
    scr_time_list.append(time.process_time() - start)
scr_time_list = pd.Series(scr_time_list)
scr_time_list.to_csv("SCC461\scr_time_list.csv", index=False)
"""














