# -*- coding: utf-8 -*-
"""Task3.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1PSjiU0MKG3u7RVZjlQIaqUiK0iPUeQgb
"""

import tensorflow as tf
from tensorflow.keras import datasets,layers, models
import matplotlib.pyplot as plt
import numpy as np

(Xtrain,ytrain),(Xtest,ytest) = datasets.cifar10.load_data()
Xtrain.shape

Xtest.shape

ytrain[:5]
#these outputs are 2D arrays but we just need direct clses so we reshape it

ytrain=ytrain.reshape(-1,)
ytrain[:5]

cls = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

cls[9]

def pltsample(X,y,index):
  plt.figure(figsize=(15,2))
  plt.imshow(X[index])
  plt.xlabel(cls[y[index]])

pltsample(Xtrain,ytrain,0)
pltsample(Xtrain,ytrain,1)
pltsample(Xtrain,ytrain,2)
pltsample(Xtrain,ytrain,3)
pltsample(Xtrain,ytrain,4)

"""# Normalization"""

Xtrain= Xtrain/255.0
Xtest= Xtest/255.0

"""# Neural Network"""

ann = models.Sequential([
    layers.Flatten(input_shape=(32,32,3)),
    layers.Dense(3000,activation='relu'),
    layers.Dense(1000,activation='relu'),
    layers.Dense(10,activation='sigmoid')
])

ann.compile(optimizer = 'SGD',
            loss = 'sparse_categorical_crossentropy',
            metrics = ['accuracy'])

ann.fit(Xtrain,ytrain,epochs=7)

"""we are using sparse_categorical_crossentropy because we have assigned direct values and not one hot encoded vector which in case we would've used categorical_entropy

"""

ann.evaluate(Xtest,ytest)

"""So we can say that ann is performing really bad on this dataset"""

from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
y_pred = ann.predict(Xtest)
y_pred_classes = [np.argmax(element) for element in y_pred]

print("Classification Report: \n",classification_report(ytest,y_pred_classes))

"""#Convolutional Neural Networks
Now we could say that seeing the results the ann didnot perform well on this dataset

so, to improve the performance we use **CNN**
"""

from tensorflow.keras import layers, models, callbacks

model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3), padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3,3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    layers.Dropout(0.3),

    layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    layers.Dropout(0.4),

    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

"""sigmoid we dont necessarily get 1 as the sum
lets say 1:0.45 and 2:0.67

but using softmax normalises the probability
1: (0.45)/(0.45+0.67) = 0.40
2: (0.67)/(0.45+0.67) = 0.60
sum is 1
"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    horizontal_flip=True,
    zoom_range=0.2
)
datagen.fit(Xtrain)

opt = tf.keras.optimizers.Adam(learning_rate=0.0008)

model.compile(optimizer=opt,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)

history = model.fit(datagen.flow(Xtrain, ytrain, batch_size=64),
                    validation_data=(Xtest, ytest),
                    epochs=20,
                    callbacks=[early_stop])

model.save("cifar10_model.keras")