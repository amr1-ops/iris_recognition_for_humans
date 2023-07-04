# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 16:54:02 2021

@author: mohamed
"""





import numpy as np
import os
import cv2
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import layers,models


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
                test_data.append([img_array, class_num])  # add this to our training_data
            except Exception as e:  # in the interest in keeping the output clean...
                pass
            
create_training_data()
create_test_data()



X = []
y = []
Xtest =[]
ytest =[]

for features,label in training_data:
    X.append(features)
    y.append(label)

for features,label in test_data:
    Xtest.append(features)
    ytest.append(label)


X = np.array(X)#.reshape(-1, 320, 280, 1)
y=np.array(y)

Xtest = np.array(X)#.reshape(-1, 320, 280, 1)
ytest=np.array(y)

X=X/255.0
Xtest=Xtest/255.0



model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3),activation='relu', input_shape=(400,250,3)))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(64, (3, 3),activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

#--------------------------------------------

model.add(layers.Dropout(0.1))

model.add(layers.Flatten())  # this converts our 3D feature(width,height,channel) maps to 1D feature vectors

model.add(layers.Dense(64,activation='relu'))#feature selection

model.add(layers.Dense(5))#activition = "softmax"


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])#combination between two GD methodologies "adam"


history = model.fit(X, y,epochs=15, validation_data=(Xtest, ytest))
test_loss, test_acc = model.evaluate(Xtest,  ytest, verbose=2)
print("accuracy ",test_acc)

#model.save("iris recognition.model")


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
