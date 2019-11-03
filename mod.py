# Made by Kshitiz and Chinmay

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from tensorflow import keras
import cv2
import random
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import scipy
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.datasets import cifar10


IMG_SIZE = 299


''' Trianing Data Directory '''
''' Trianing Data Directory '''
''' Trianing Data Directory '''
TRAIN_DIREC = "/content/drive/My Drive/horses-or-humans-dataset/horse-or-human/train"
CATEGORIES = ["horses", "humans"]


train_datagen = ImageDataGenerator(
    
    horizontal_flip=.5,
    fill_mode='nearest')

training_data = []
test_data = []
def create_training_data():
    for category in CATEGORIES:
      path = os.path.join(TRAIN_DIREC, category)

      class_num = CATEGORIES.index(category)
      for img in tqdm(os.listdir(path)):
        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
        img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE)).reshape(IMG_SIZE,IMG_SIZE,1)
        img_array = np.expand_dims(img_array,0)
        aug_iter = train_datagen.flow(img_array)
        aug_images = [next(aug_iter)[0].astype(np.uint64) for i in range(11)]
        labels = [class_num for i in range(11)]
        training_data.append([aug_images, labels])
   
       


create_training_data()

random.shuffle(training_data)
testing_data = []




''' Testing Data directory '''
''' Testing Data directory '''
''' Testing Data directory '''
TEST_DIREC = "/content/drive/My Drive/horses-or-humans-dataset/horse-or-human/validation"
def create_testing_data():
  for category in CATEGORIES:
      path = os.path.join(TEST_DIREC, category)

      class_num = CATEGORIES.index(category)
      for img in tqdm(os.listdir(path)):
        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
        img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
        testing_data.append([img_array, class_num])

create_testing_data()

X = []
y = []

for features, label in training_data:
    for xa in features:
      X.append(xa)
    for ya in label:
      y.append(ya)
X = np.array(X).reshape(-1,IMG_SIZE, IMG_SIZE,1)
y = np.array(y)
X = X/255.0

X_test = []
y_test = []

for features, label in testing_data:
  X_test.append(features)
  y_test.append(label)

X_test = np.array(X_test).reshape(-1,IMG_SIZE, IMG_SIZE,1)
X_test = X_test/255.0
y_test = np.array(y_test)

model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))


epochs = 10
model.add(Flatten())

model.add(Dense(64, activation = 'relu'))
model.add(Dense(1))
model.add(Activation('sigmoid'))


model.compile(loss ='binary_crossentropy',optimizer = 'sgd' ,metrics =['accuracy'])
model.fit(X, y, epochs = 10)
predd, acc = model.evaluate(X_test, y_test)
print(acc)

