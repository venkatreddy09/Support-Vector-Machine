# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 12:15:01 2019

@author: Venkat Reddy
"""

#import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt

#importing dataset
dataset=pd.read_csv('Social_Network_Ads.csv')
x=dataset.iloc[:,[2,3]].values
y=dataset.iloc[:,4].values

#splitting the dataset into training and test set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_tarin=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

#Create the model for SVM
from sklearn.svm import SVC
classifier=SVC(kernel='linear',random_state=0)
classifier.fit(x_train,y_train)

#predict the test set results
y_predict=classifier.predict(x_test)

#making the confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_predict)

