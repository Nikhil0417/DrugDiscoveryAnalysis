import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

df1 = pd.read_csv("adrb2-properties_cleaned.csv")

df2 = df1.iloc[:,1:52]
#print(df2)

df3 = pd.read_csv("adrb2_binding.csv")
y_df = df3.iloc[:,5]

y_df.fillna(0, inplace=True)
z = np.array(y_df.replace('Y', 1))

print(df3.shape)
print(z)
print(type(z))

#Data pre-processing	NORMALIZATION
x = df2.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x_scaled,z,test_size=0.25, random_state=0)

print(x_train.shape)
print(x_test.shape) #has 115 magic numbers
print(y_train.shape)
print(y_test.shape) #has 41 magic numbers
print(type(x_train))
print(type(x_test))
print(type(y_train))
print(type(y_test))

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

#create adaboosted DT
adaBoostedTree = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
									algorithm="SAMME.R",
									n_estimators=200)

adaBoostedTree.fit(x_train,y_train)
result = adaBoostedTree.predict(x_test)

print(result)
print("---------")
print(y_test)

print("score:", adaBoostedTree.score(x_test,y_test))
print("Predicted probabilities:", adaBoostedTree.predict_proba(x_test))
