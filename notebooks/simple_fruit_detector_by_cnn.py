#!/usr/bin/env python
# coding: utf-8

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout, GlobalMaxPooling2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.models import Model
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import numpy as np
import pandas as pd


data_dir_name = '../datasets/'


def train_model():
    # With Augmentation
    ds = ImageDataGenerator(zoom_range=0.2,
                           horizontal_flip=True,
                           rescale = 1/255, 
                           validation_split = 0.3)

    img_height = 400
    img_width = 400

    train_dataset = ds.flow_from_directory(directory = data_dir_name,
                                          subset='training',
                                          target_size = (img_height ,img_width),
                                          class_mode = 'sparse'
                                          )

    validation_dataset = ds.flow_from_directory(directory = data_dir_name,
                                              subset='validation',
                                              target_size = (img_height ,img_width),
                                              class_mode = 'sparse'
                                               )


    class_names = list(train_dataset.class_indices)
    num_classes = len(set(class_names))
    print(f'class names:{class_names}')
    print(f'num of classes:{num_classes}')

    (a, b) = train_dataset[0]

    i = Input(shape=a[0].shape)
    # x = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)(i) # standartization layer
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(i)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.2)(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.2)(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.2)(x)
    x = GlobalMaxPooling2D()(x)

    x = Flatten()(x)
    x = Dropout(0.2)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(num_classes, activation='softmax')(x)

    model = Model(i, x)


    model.compile(
      optimizer='RMSprop',
      loss='sparse_categorical_crossentropy',
      metrics=['accuracy'])


    checkpoint = ModelCheckpoint("../models/cnn-model.h5", monitor='loss', verbose=1, save_weights_only=True, mode='auto')
    
    model.fit(
        train_dataset,
        validation_data=validation_dataset,
        batch_size = 32,
        epochs=5,
#         callbacks=[checkpoint]
    )

    model.save("../models/CNN-output.h5")

    return model





