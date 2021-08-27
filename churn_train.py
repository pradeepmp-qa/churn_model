import tensorflow
tensorflow.__version__

# importing of all the important libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# tesnorflow lib
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense , Dropout
import os

cwd = os.getcwd() ##gets current working directory

data_path = str(cwd) + '/Churn_Modelling.csv'

data = pd.read_csv(data_path)  ## Loads the data


print(data)

data.info()

# split X and Y

X = data.iloc[: , 3:13]
y = data.iloc[: , 13]

X

y

geography = pd.get_dummies(X['Geography'] , drop_first=True)
gender = pd.get_dummies(X['Gender'] , drop_first=True)

X = pd.concat([X , geography, gender ] , axis = 1)

X

X = X.drop(columns = ['Geography' , 'Gender'] , axis = 1)

# Spillting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y , test_size = 0.2 , random_state = 101)

# feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train.shape

#1. architecture creation
#2. compile
#3 . model train

#1. model creation

model = Sequential()

model.add(Dense(200, activation='relu' , kernel_initializer='he_uniform')) # hidden layer 1
model.add(Dropout(0.2))

model.add(Dense(50 , activation='sigmoid')) # hidden layer 2
model.add(Dropout(0.5))

model.add(Dense(25 , activation='relu' , kernel_initializer='he_normal')) # hidden layer 3

model.add(Dense(1 , activation='sigmoid')) # output layer

model.compile(optimizer='SGD' , loss = 'binary_crossentropy' , metrics=['accuracy'])

model.fit(X_train , y_train , epochs = 10 , batch_size=32)

# testing the data
model.evaluate(X_test , y_test )


import joblib

joblib.dump(sc , 'feature_sacling.pkl')
model.save('20_aug_82.h5')

# loading of the model

from keras.models import load_model
model_load = load_model('20_aug_82.h5')

model_load.predict(X_test)


y_pred = model_load.predict(X_test) > 0.5
y_pred



# a = 10
# b = 10
# c = 1000
# d = 10
# e = 10
# f = 10
# g = 1000
# h = 10
# i = 10
# j = 1000
# k = 10

# scaling = joblib.load('feature_sacling.pkl')

# data = scaling.transform([[a,b,c,d,e,f,g,h,i,j,k]])

# print(data)

# output = model_load.predict(data) > 0.5

# if output[0][0] == True:
#   print('he\she will leave')
# else:
#   print('he/she will stay')
