# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 16:17:38 2021

@author: mohamed
"""



import os

import matplotlib as mpl
import matplotlib.pyplot as plt
from IPython.display import display

import pandas as pd
import numpy as np

from PIL import Image

from skimage.feature import hog
from skimage.color import rgb2grey

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC

from sklearn.metrics import roc_curve, auc

import cv2
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import datasets,layers,models

DATADIR = "dataset/train"
TESTDIR = "dataset/test"

CATEGORIES = ["Ahmed Amr","Ali Habib","Mohamed Bebo","Mohamed Labib","Mohamed Mokhtar"]

training_data = []
test_data = []

def create_training_data():
    for category in CATEGORIES:  # do

        path = os.path.join(DATADIR,category)  # create path 
        class_num = CATEGORIES.index(category)  # get the classification

        for img in tqdm(os.listdir(path)):  # iterate over each image
            try:
                img_array = cv2.imread(os.path.join(path,img))  # convert to array
                img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB) #it may be not necessory but i put it as a precaution
                img_array=cv2.resize(img_array, (400, 250))
                img_array = hog(img_array, block_norm='L1', pixels_per_cell=(16, 16))
                img_array = np.array(img_array)
                training_data.append([img_array, class_num])  # add this to our training_data
            except Exception as e:  # in the interest in keeping the output clean...
                pass
            
def create_test_data():
    for category in CATEGORIES:  # do

        path = os.path.join(TESTDIR,category)  # create path 
        class_num = CATEGORIES.index(category)  # get the classification

        for img in tqdm(os.listdir(path)):  # iterate over each image
            try:
                img_array = cv2.imread(os.path.join(path,img))  # convert to array
                img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
                img_array=cv2.resize(img_array, (400, 250))
                img_array = hog(img_array, block_norm='L1', pixels_per_cell=(16, 16))
                img_array = np.array(img_array)
                test_data.append([img_array, class_num])  # add this to our training_data
            except Exception as e:  # in the interest in keeping the output clean...
                pass
            
create_training_data()
create_test_data()



       




ytrain = []
Xtrain = []
ytest = []
Xtest = []



for feature,label in training_data:
    ytrain.append(label)
    Xtrain.append(feature)  

for feature,label in test_data:
    ytest.append(label)
    Xtest.append(feature)  






Xtrain = np.array(Xtrain)
Xtest = np.array(Xtest) 
ytrain = np.array(ytrain)
ytest = np.array(ytest)



print("Start training..\n")

model = models.Sequential()

model.add(layers.Dense(64,activation='relu'))

model.add(layers.Dropout(0.1))


model.add(layers.Dense(64,activation='relu'))#feature selection

model.add(layers.Dense(5))#activition = "softmax"


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])#combination between two GD methodologies "adam"


history = model.fit(Xtrain, ytrain,epochs=15, validation_data=(Xtest, ytest))
test_loss, test_acc = model.evaluate(Xtest,  ytest, verbose=2)
print("accuracy ",test_acc)#99

#model.save("ann.model")#1.0

print("End training..\n")

import matplotlib.pyplot as plt 

#loss curve
loss_train = history.history['loss']
epochs = range(1,16)
plt.plot(epochs, loss_train, 'g', label='Training loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#accuracy curve
loss_train = history.history['accuracy']
epochs = range(1,16)
plt.plot(epochs, loss_train, 'g', label='Training accuracy')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
