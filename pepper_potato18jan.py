# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 12:59:18 2021

@author: ocn
"""
#increased epochs from 10 to 20 and changed modelname to model....2.h5
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.preprocessing import image

# for plotting images (optional)
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
    plt.tight_layout()
    plt.show()
    

# getting data
base_dir = 'E:/AVRN_Report/PlantVillage/pepper_potato'
train_dir = os.path.join(base_dir, 'train')
valid_dir = os.path.join(base_dir, 'valid')

train_pepper_Bacterial_spot = os.path.join(train_dir, 'pb_bacterial')
train_pepper_healthy = os.path.join(train_dir, 'pb_healthy')
train_potato_eb = os.path.join(train_dir, 'potato_eb')
train_potato_healthy = os.path.join(train_dir, 'potato_healthy')
train_potato_lb = os.path.join(train_dir, 'potato_lb')

valid_pepper_Bacterial_spot = os.path.join(valid_dir, 'pb_bacterial')
valid_pepper_healthy = os.path.join(valid_dir, 'pb_healthy')
valid_potato_eb = os.path.join(valid_dir, 'potato_eb')
valid_potato_healthy = os.path.join(valid_dir, 'potato_healthy')
valid_potato_lb = os.path.join(valid_dir, 'potato_lb')

num_pepper_Bacterial_spot_tr = len(os.listdir(train_pepper_Bacterial_spot))
num_pepper_healthy_tr = len(os.listdir(train_pepper_healthy))
num_potato_eb_tr = len(os.listdir(train_potato_eb))
num_potato_healthy_tr = len(os.listdir(train_potato_healthy))
num_potato_lb_tr = len(os.listdir(train_potato_lb))

num_pepper_Bacterial_spot_vl = len(os.listdir(valid_pepper_Bacterial_spot))
num_pepper_healthy_vl = len(os.listdir(valid_pepper_healthy))
num_potato_eb_vl = len(os.listdir(valid_potato_eb))
num_potato_healthy_vl = len(os.listdir(valid_potato_healthy))
num_potato_lb_vl = len(os.listdir(valid_potato_lb))


total_train = num_pepper_Bacterial_spot_tr + num_pepper_healthy_tr + num_potato_eb_tr + num_potato_healthy_tr + num_potato_lb_tr
total_val = num_pepper_Bacterial_spot_vl +  num_pepper_healthy_vl + num_potato_eb_vl + num_potato_healthy_vl + num_potato_lb_vl

BATCH_SIZE = 32
IMG_SHAPE = 200 # square image


#generators

#prevent memorization
train_image_generator = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
    )

validation_image_generator = ImageDataGenerator(
    rescale=1./255)


train_data_gen = train_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_SHAPE, IMG_SHAPE),
                                                           class_mode='binary')

val_data_gen = train_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                           directory=valid_dir,
                                                           shuffle=False,
                                                           target_size=(IMG_SHAPE, IMG_SHAPE),
                                                           class_mode='binary')
images = [train_data_gen[0][0][0] for i in range(5)]
plotImages(images)


model = Sequential()
# Conv2D : Two dimentional convulational model.
# 32 : Input for next layer
# (3,3) convulonational windows size
model.add(Conv2D(32, (3, 3), input_shape=(IMG_SHAPE, IMG_SHAPE,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.5)) # 1/2 of neurons will be turned off randomly
model.add(Flatten())
model.add(Dense(256, activation='relu'))

# output dense layer; since thenumbers of classes are 5 here so we need to pass minimum 5 neurons whereas 2 in cats and dogs   
model.add(Dense(5, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()


EPOCHS = 20

history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=int(np.ceil(total_train / float(BATCH_SIZE))),
    epochs=EPOCHS,
    validation_data=val_data_gen,
    validation_steps=int(np.ceil(total_val / float(BATCH_SIZE)))
    )


# analysis
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

model.save("model_pepper_potato_plant_disease2.h5")
print("Saved model to disk")