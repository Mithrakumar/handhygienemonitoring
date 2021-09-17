#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  26 12:32:10 2021

@author: Mithra
"""

import pandas as pd
import os
import numpy as np
import keras
import tensorflow as tf
tf.autograph.set_verbosity(3, True)

# The input of CNN (square image)
IMAGE_SIZE = 128
BATCH_SIZE = 4
test_images_dir = "/Users/Mithra/Desktop/Hand Hygiene Monitoring/test_final"
train_images_dir= "/Users/Mithra/Desktop/Hand Hygiene Monitoring/train_final"
sample_sub_path = "/Users/Mithra/Desktop/Hand Hygiene Monitoring/Kaggle spreadsheet WHO - final.csv"
# Submission example
submit_example = pd.read_csv(sample_sub_path)
submit_example.head()
list_images_path = os.listdir(train_images_dir)

labels = []
for image_path in list_images_path:
    if "Step1" in image_path:
        labels.append('Step1')
    elif "Step2"in image_path:
        labels.append('Step2')
    elif "Step3" in image_path:
        labels.append('Step3')
    elif "Step4" in image_path:
        labels.append('Step4')
    elif "Step5" in image_path:
        labels.append('Step5')
    else:
        labels.append('Step6')

df = pd.DataFrame({"file":list_images_path, "label":labels})

print(df.head())
print(df.tail())
df['label'].hist()
from PIL import Image
import matplotlib.pyplot as plt
sample = np.random.randint(len(df))

img_path = train_images_dir +'/'+df['file'][sample]
label    = df['label'][sample]

img = Image.open(img_path)

plt.imshow(img)
plt.title(f"Class:{label}")
plt.axis('off')
df.shape
from sklearn.model_selection import train_test_split
train_df, valid_df = train_test_split(df, test_size = 0.15)
print("Train set:", train_df.shape)
print("Validation set:", valid_df.shape)
train_df.head()
from keras.preprocessing.image import ImageDataGenerator

train_gen = ImageDataGenerator(rescale = 1.0/255.0,
                               horizontal_flip = True,
                               vertical_flip   = True,
                               fill_mode = 'nearest',
                               rotation_range = 10,
                               width_shift_range = 0.2,
                               height_shift_range= 0.2,
                               shear_range= 0.15,
                               brightness_range= (.5,1.2),
                               zoom_range = 0.2)

# https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator#flow_from_dataframe
train_gen = train_gen.flow_from_dataframe(train_df,
                                          directory = train_images_dir,
                                          x_col = 'file', 
                                          y_col = 'label', 
                                          target_size =(IMAGE_SIZE, IMAGE_SIZE), 
                                          class_mode = 'categorical',
                                          batch_size = BATCH_SIZE, 
                                          color_mode = 'rgb', 
                                          shuffle = True)
n_samples = 8

plt.figure(figsize=(20,20))
for x_gens, y_gens in train_gen:
#     the first dimension of x_gens and y_gens will be equal to batch_size specifed previously
    print(x_gens.shape)
    i = 0
    for sample_img, sample_class in zip(x_gens, y_gens):
        
        plt.subplot(5,4,i+1)
        plt.title(f'Class:{np.argmax(sample_class)}')
        plt.axis('off')
        plt.imshow(sample_img)
        
        i += 1
        
        if i >= n_samples:
            break
    break
valid_gen = ImageDataGenerator(rescale=1./255)
valid_gen = valid_gen.flow_from_dataframe( valid_df, 
                                           directory = train_images_dir,
                                           x_col='file',
                                           y_col='label',
                                           target_size=(IMAGE_SIZE,IMAGE_SIZE),
                                           class_mode='categorical',
                                           batch_size=BATCH_SIZE)
from tensorflow import keras
def myModel(input_shape):
    X_input = keras.layers.Input(shape=input_shape, name='input')
    
#     128x128x3
    
    X = keras.layers.Conv2D(filters=32, kernel_size=(3,3), padding='same', name='conv-1')(X_input)    
    X = keras.layers.BatchNormalization(name='b1')(X)
    X = keras.layers.Activation('relu')(X)
    X = keras.layers.MaxPool2D(pool_size=(2,2))(X)
    X = keras.layers.Dropout(0.2)(X)
    
#     64x64x32
    
    X = keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding='same', name='conv-2')(X)    
    X = keras.layers.BatchNormalization(name='b2')(X)
    X = keras.layers.Activation('relu')(X)
    X = keras.layers.MaxPool2D(pool_size=(2,2))(X)
    X = keras.layers.Dropout(0.2)(X)
#     32x32x64

    X = keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding='same', name='conv-3')(X)    
    X = keras.layers.BatchNormalization(name='b3')(X)
    X = keras.layers.Activation('relu')(X)
    X = keras.layers.MaxPool2D(pool_size=(2,2))(X)
    X = keras.layers.Dropout(0.2)(X)

#     16x16x128
    
    X = keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding='same', name='conv-4')(X)    
    X = keras.layers.BatchNormalization(name='b4')(X)
    X = keras.layers.Activation('relu')(X)
    X = keras.layers.MaxPool2D(pool_size=(2,2))(X)
    X = keras.layers.Dropout(0.2)(X)

#     8x8x128
    X = keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding='same', name='conv-5')(X)    
    X = keras.layers.BatchNormalization(name='b5')(X)
    X = keras.layers.Activation('relu')(X)
    X = keras.layers.MaxPool2D(pool_size=(2,2))(X)
    X = keras.layers.Dropout(0.2)(X)
    
#     8x8x128
    
    X = keras.layers.Flatten()(X)
    X = keras.layers.Dense(units=512, name='fc-6')(X)
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.Activation('relu')(X)
    X = keras.layers.Dropout(0.3)(X)
    
    X = keras.layers.Dense(units=6, activation='softmax', name='output')(X)
    
    model = keras.Model(inputs = X_input, outputs = X, name='My_CNN_Model')
    
    return model
model = myModel((IMAGE_SIZE,IMAGE_SIZE,3))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
# To stop the training after N epochs and val_loss value not decreased
earlystop = EarlyStopping(patience=2)
# To reduce the learning rate when the accuracy not increase for 5 steps
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=2, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.0001)
# callbacks = [earlystop, learning_rate_reduction]
callbacks = [earlystop]
epochs = 30
history = model.fit(train_gen, 
                    steps_per_epoch = len(train_df)//BATCH_SIZE, 
                    epochs = epochs, 
                    validation_data = valid_gen, 
                    validation_steps = len(valid_df)//BATCH_SIZE)
history.history.keys()
print("Accuracy = ", history.history['accuracy'][-1])
print("Val. Accuracy = ", history.history['val_accuracy'][-1])

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1,len(loss)+1)

plt.plot(epochs, loss, label='Train set')
plt.plot(epochs, val_loss, label='Validation set')

plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
epochs = range(1,len(loss)+1)

plt.plot(epochs, accuracy, label='Train set')
plt.plot(epochs, val_accuracy, label='Validation set')

plt.legend()
plt.xlabel('Accuracy')
plt.ylabel('Loss')
plt.show()

test_images_path = os.listdir(test_images_dir)
#test_df = pd.DataFrame({'file':test_images_path})
img_path = test_images_dir +'/'+df['file']
test_df1= pd.DataFrame({'file':img_path})
test_df1.head()

test_gen = ImageDataGenerator(rescale=1.0/255.0)
test_gen = test_gen.flow_from_dataframe(test_df1, 
                                        directory=test_images_dir, 
                                        x_col='file', 
                                        y_col=None,
                                        class_mode=None,
                                        target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                        color_mode="rgb",
                                        shuffle = False)
predictions = model.predict(test_gen)
predictions = np.argmax(predictions,axis=1)
predictions.shape
nsamples = 8

print("Predictions: ", predictions)

def check_availability(element, collection: iter):
    return element in collection
if check_availability('0', predictions) and check_availability('1', predictions) and check_availability('2', predictions) and check_availability('3', predictions) and check_availability('4', predictions) and check_availability('5', predictions):
    print("Entry granted")
else:
    print("Entry denied")
    
    
#plt.figure(figsize=(20,20))
#for i, file in enumerate(img_path['file'][:nsamples]):
#for i, file in enumerate(img_path[:nsamples]):
    #img = Image.open(test_images_dir+file)
    
    #plt.subplot(5,4, i+1)
    #plt.imshow(img)
    #plt.title(f"Class:{predictions[i]}")
    #plt.axis('off')

submit_df = pd.DataFrame()
submit_df['id'] = range(1,len(predictions)+1)
submit_df['label'] = predictions

submit_df.to_csv('WHO_1.csv', index=False)

submit_df.head(10)
