
# coding: utf-8

# In[24]:


import numpy as np
from sklearn import svm 
from sklearn import preprocessing
from math import sqrt
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import pandas as pd
import csv
import glob
import os
import copy               # importing needed library for SVM and data processing 


# In[60]:


dataPath = "/Users/ASDERTY/Documents/date IA/"
path = '/Users/ASDERTY/Documents/date IA/train/'
pathTest = '/Users/ASDERTY/Documents/date IA/test/'
testedFile = []
o = 0
train_images = []
test_images = []
#np.concatenate((test_images[x],[[0,0,0],[0,0,0]]),axis = 0)
train_labels = np.array(pd.read_csv("/Users/ASDERTY/Documents/date IA/"+ "train_labels.csv", header=None))
for root,dirs,files in os.walk(path):
    for x in files:
        train_image = (np.array(pd.read_csv(path + x, header=None)))
        for i in range(train_image.shape[0],156):                                         # using panda format for interpolation
            train_image= np.concatenate((train_image,[[np.nan,np.nan,np.nan]]),axis = 0)  #because the files size varies between
        train_image = pd.Series(train_image.flatten())                                    #136-156 and entered interpolated data
        train_image = train_image.interpolate()
        train_image = np.array(train_image)
        train_image = train_image.flatten()
        train_images.append((train_image))
o = 0
for root,dirs,files in os.walk(pathTest):
    testedFiles = files
    for x in files:
        test_image = (np.array(pd.read_csv(pathTest + x, header=None)))
        for i in range(test_image.shape[0],159):
            test_image= np.concatenate((test_image,[[np.nan,np.nan,np.nan]]),axis = 0)    #same as above, only that the test files
        test_image = pd.Series(test_image.flatten())                                    #have a max value of 159 of data
        test_image = test_image.interpolate()
        test_image  = np.array(test_image)
        test_image=  test_image.flatten()
        test_images.append((test_image))

train_images = np.array(train_images)
test_images = np.array(test_images)
print(train_images.shape)
print(test_images.shape)
filesGood = []
for x in testedFiles:
    filesGood.append(x.replace('.csv',''))


# In[61]:



print(type(train_images[0][0]))
print(train_images[0])
plt.plot(train_images[0],'ro')
plt.plot(train_images[1],'bo')
plt.plot(train_images[2],'go')         #checking the data and plot it to see a pattern on the train_data
plt.plot(train_images[3],'co')
plt.show()


# In[62]:


def normalize_data(test_data,type):
    if type == 'standard':
        scaler = preprocessing.StandardScaler()       #methods of normalization of data
        scaler.fit(test_data)
        scaled_test_data = scaler.transform(test_data)
    elif type == 'min_max':
        scaler = preprocessing.MinMaxScaler()
        scaler.fit(test_data)
        scaled_test_data = scaler.transform(test_data)
    elif type == "l1":
        scaled_test_data = test_data/np.expand_dims(np.sum(abs(test_data),axis = 0),axis = 0)
    elif type == "l2":
        scaled_test_data = test_data/np.expand_dims(np.sqrt(np.sum(test_data**2,axis = 0)), axis = 0 )
    return scaled_test_data


# In[63]:


scaled_data = copy.deepcopy(train_images)
scaled_test = copy.deepcopy(test_images)
scaled_data = normalize_data(scaled_data,'standard')           #making a deep copy to have it unique to normalize
scaled_test = normalize_data(scaled_test,'standard')
print(scaled_data)
plt.plot(scaled_data[0],'ro')
plt.plot(scaled_data[2],'go')
plt.plot(scaled_data[3],'co')
plt.show()


# In[64]:


def svm_classifier(train_data,train_labels,svm_model,test_data):
    print(type(train_data))                                     #define a svm classifier to predict and send continous data
    print(type(svm_model))
    svm_model.fit(train_data,train_labels)
    predicted_labels = svm_model.predict(test_data)
    return (predicted_labels,svm_model)


# In[65]:


scaled_train_data = scaled_data
scaled_test_data=scaled_test


# In[66]:


accuracy_test= []
accuracy_train = []           #lookign for the best value , keept the default kernel for mutiple point clasification on 3D space
best_c = [1,5,10,20,30]
for x in best_c:
    svm_model =svm.SVC(C=x)
    pred_x,svm_model=svm_classifier(scaled_train_data[:7000],train_labels[:7000,[1]].flatten(),svm_model,scaled_train_data)
    print(pred_x)
    print(metrics.accuracy_score(train_labels[7000:9000,[1]],pred_x[7000:9000]),'   *************   ')
    print(metrics.accuracy_score(train_labels[0:7000,[1]],pred_x[0:7000]),'   *************   ')


# In[45]:


conf_matrix=metrics.confusion_matrix(train_labels[4500:9000,[1]],pred_y)             #verifing preccision,recall,and confusion metrix to see the respective recurenace and percentaje
precission_per_class = [conf_matrix[x][x]/np.sum(conf_matrix[:,x]) for x in range(0,20)]
recall_per_class = [conf_matrix[x][x]/(np.sum(conf_matrix[x,:])+conf_matrix[x][x])for x in range(0,20)]
print(pred_x)
print(metrics.confusion_matrix(train_labels[4500:9000,[1]],pred_x))
print(metrics.confusion_matrix(train_labels[:4500,[1]],pred_x))
print(accuracy_test)
print(precission_per_class)
print(recall_per_class)


# In[ ]:


predict_final = svm_model.predict(scaled_test_data) #getting the final data predicted


# In[12]:


print(predict_final)
print(dataPath)
print(len(predict_final))
with open(dataPath + 'prezic.csv', mode='w', newline='') as sm: #creating a file for the test data 
    writer = csv.writer(sm, delimiter=',')
    writer.writerow(['id', 'class'])
    for i in range(5000):
        writer.writerow([filesGood[i],predict_final[i]])

