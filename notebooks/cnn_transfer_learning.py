#!/usr/bin/env python
# coding: utf-8

import keras_preprocessing
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from keras.models import load_model

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from joblib import dump,load


def train_model():
    directory = './datasets/'

    train_dir = valid_dir = directory

    train_datagen = ImageDataGenerator(rescale = 1./255,
                                      rotation_range=40,
                                      width_shift_range=0.2,
                                      height_shift_range=0.2,
                                      shear_range=0.2,
                                      zoom_range=0.2,
                                      horizontal_flip=True,
                                      fill_mode='nearest',
                                      validation_split=0.2)


# valid_datagen = ImageDataGenerator(rescale = 1./255)

    train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(200,200),
                                                    class_mode='categorical',
                                                    batch_size=5,
                                                    subset='training')

    valid_generator = train_datagen.flow_from_directory(valid_dir,
                                                    subset='validation',
                                                    target_size=(200,200),
                                                    class_mode='categorical',
                                                    batch_size=5,
                                                    )
    
    class_names = list(train_generator.class_indices)
    num_classes = len(set(class_names))

    vgg_model = VGG16(include_top=False, input_shape=(200, 200, 3))

    for layer in vgg_model.layers:
        layer.trainable = False

    flat1 = Flatten()(vgg_model.layers[-1].output)
    class1 = Dense(256, activation='relu')(flat1)
    output = Dense(num_classes, activation='softmax')(class1)

    model = Model(inputs = vgg_model.inputs, outputs = output)
    
    model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    history = model.fit(train_generator,
                    validation_data = valid_generator,
                    epochs=5, verbose=1)

    # model.save("../models/CNN-output.h5")
    out = []
    res = []

    for val in range(len(valid_generator)):
        (a, b) = valid_generator[val]
        out.append(a)
        res.append(b)

    print(len(out), len(res))
    o = out[0]
    r = res[0]

    for i in range(1, len(out)):
        o = np.concatenate((o, out[i]))
        r = np.concatenate((r, res[i]))

    pred = model.predict(o)
    pred = [np.argmax(val) for val in pred]
    r = [np.argmax(val) for val in r]
    
    cm = confusion_matrix(r, pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cmd = ConfusionMatrixDisplay(cm, display_labels=class_names)

    cmd.plot()
    cmd.ax_.set(title='Confusion Matrix',
            xlabel='Predicted Fruits',
            ylabel='Actual Fruits')

    plt.savefig('./cnn.jpg')
    
    return model

def predict_cnn_model(data):
    directory = './datasets/'

    train_dir = valid_dir = directory

    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       rotation_range=40,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True,
                                       fill_mode='nearest',
                                       validation_split=0.2)

    # valid_datagen = ImageDataGenerator(rescale = 1./255)

    train_generator = train_datagen.flow_from_directory(train_dir,
                                                        target_size=(200, 200),
                                                        class_mode='categorical',
                                                        batch_size=5,
                                                        subset='training')

    class_names = list(train_generator.class_indices)
    num_classes = len(set(class_names))
    class_names = ['apple', 'broccoli', 'grape', 'lemon', 'orange', 'strawberry']
    print(class_names, num_classes)
    data = [data]
    data = np.array(data)
    print(data.shape)

    model = load_model('./CNN-output.h5')
    pred = model.predict(data)

    print('predictions=', pred)
    predictions = class_names[np.argmax(pred)]
    print('final value', predictions)

    return predictions


