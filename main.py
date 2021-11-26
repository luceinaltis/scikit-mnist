import matplotlib.pyplot as plt
import time

from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import tensorflow as tf 

import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU

dataset = datasets.load_digits()
x_data = dataset.data
y_data = dataset.target

img_rows, img_cols=8, 8

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3)

x_train = x_train.reshape(-1, 8,8, 1)
x_test = x_test.reshape(-1, 8,8, 1)
print(x_train.shape)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = x_train / 255.
x_test = x_test / 255.

train_Y_one_hot = tf.keras.utils.to_categorical(y_train)
test_Y_one_hot = tf.keras.utils.to_categorical(y_test)

model = Sequential()
model.add(Conv2D(4, kernel_size=(3, 3),activation='linear',input_shape=(8,8,1),padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D((2, 2),padding='same'))
model.add(Flatten())
model.add(Dense(36, activation='linear'))
model.add(LeakyReLU(alpha=0.1))                  
model.add(Dense(10, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=tf.optimizers.Adam(),metrics=['accuracy'])

model.fit(x_train, train_Y_one_hot, epochs=3000, batch_size=512)


start_time = time.process_time_ns()
score = model.evaluate(x_test, test_Y_one_hot, verbose=0)
end_time = time.process_time_ns()
print("Logistic Regression: ", end_time - start_time, "ns")

print('loss=', score[0])
print('accuracy=', score[1])