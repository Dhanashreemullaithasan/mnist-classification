# Convolutional Deep Neural Network for Digit Classification

## AIM

To Develop a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images.

## Problem Statement and Dataset
Classify the digits and confirm the answer for scanned handwritten pictures.

The handwritten numbers of the MNIST dataset are gathered together. The assignment is to categorize a provided image of a handwritten digit into one of ten classes, which together represent the integer values 0 through 9. A set of 60,000 handwritten, 28 X 28 digits make up the dataset. Here, we construct a model of a convolutional neural network that can categorize to the correct numerical value.

1![images](https://github.com/user-attachments/assets/54f97716-a611-4878-9d44-fa3dd772466a)

## Neural Network Model

![367160993-7a54ea9c-fcd4-4941-93fd-f71c80e39aa7](https://github.com/user-attachments/assets/a6b2f5f8-b392-4d36-bfb3-7782a6ebd5b7)

## DESIGN STEPS
### STEP 1:
Import tensorflow and preprocessing libraries.

### STEP 2:
Download and load the dataset and Scale the dataset between it's min and max values

### STEP 3:
Using one hot encoding, encode the categorical values and Split the data into train and test

### STEP 4:
Build the convolutional neural network model

### STEP 5:
Train the model with the training data and Plot the performance plot

### STEP 6:
Evaluate the model with the testing data

### STEP 7:
Fit the model and predict the single input

## PROGRAM

### Name: DHANASHREE M
### Register Number:212221230018
```
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import utils
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.preprocessing import image


(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train.shape

X_test.shape

single_image= X_train[0]

single_image.shape

plt.imshow(single_image,cmap='gray')

y_train.shape

X_train.min()

X_train.max()

X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0

X_train_scaled.min()

X_train_scaled.max()

y_train[0]

y_train_onehot = utils.to_categorical(y_train,10)
y_test_onehot = utils.to_categorical(y_test,10)

type(y_train_onehot)

y_train_onehot.shape

single_image = X_train[500]
plt.imshow(single_image,cmap='gray')

y_train_onehot[500]

X_train_scaled = X_train_scaled.reshape(-1,28,28,1)
X_test_scaled = X_test_scaled.reshape(-1,28,28,1)

print("NAME : DHANASHREE M")
print("REG NO : 212221230018")

model = keras.Sequential()
model.add(layers.Conv2D(filters=32,kernel_size=(3,3),input_shape=(28,28,1),activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(128,activation='relu'))
model.add(layers.Dense(10,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.summary()

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(X_train_scaled ,y_train_onehot, epochs=5,
          batch_size=64, 
          validation_data=(X_test_scaled,y_test_onehot))
metrics = pd.DataFrame(model.history.history)

metrics.head()

metrics[['accuracy','val_accuracy']].plot()

metrics[['loss','val_loss']].plot()

x_test_predictions = np.argmax(model.predict(X_test_scaled), axis=1)

print(confusion_matrix(y_test,x_test_predictions))

print(classification_report(y_test,x_test_predictions))

img = image.load_img('num.png')

type(img)

img = image.load_img('num.png')
img_tensor = tf.convert_to_tensor(np.asarray(img))
img_28 = tf.image.resize(img_tensor,(28,28))
img_28_gray = tf.image.rgb_to_grayscale(img_28)
img_28_gray_scaled = img_28_gray.numpy()/255.0

x_single_prediction = np.argmax(
    model.predict(img_28_gray_scaled.reshape(1,28,28,1)),
     axis=1)

print(x_single_prediction)

plt.imshow(img_28_gray_scaled.reshape(28,28),cmap='gray')

img_28_gray_inverted = 255.0-img_28_gray
img_28_gray_inverted_scaled = img_28_gray_inverted.numpy()/255.0

x_single_prediction = np.argmax(
    model.predict(img_28_gray_inverted_scaled.reshape(1,28,28,1)),
     axis=1)

print(x_single_prediction)
```
## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot

![I1](https://github.com/user-attachments/assets/2d57e60a-6f6c-4275-9558-b9b7ced49bfb)

![I2](https://github.com/user-attachments/assets/8e596e0f-4bbb-4fff-aefe-4591174cc5bd)

![I3](https://github.com/user-attachments/assets/1ad94f68-3f13-4707-bf46-ccf422a27584)

### Classification Report

![I4](https://github.com/user-attachments/assets/65fc2688-a99f-4b66-811b-3429a71298a4)

![I5](https://github.com/user-attachments/assets/1997c8ff-2e5c-4bda-b944-461373212bc2)


### Confusion Matrix

![I6](https://github.com/user-attachments/assets/b9c5993a-a63a-4656-9958-2670919c4457)

### New Sample Data Prediction

![I7](https://github.com/user-attachments/assets/2d8e2bf3-446f-4cf0-a230-0d4068aa5c52)

## RESULT
A convolutional deep neural network for digit classification and to verify the response for scanned handwritten images is developed sucessfully.
