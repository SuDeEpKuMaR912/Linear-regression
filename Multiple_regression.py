import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

diabetes=datasets.load_diabetes()

#print(diabetes.keys())
# dict_keys(['data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename', 'target_filename', 'data_module'])

#print(diabetes.DESCR)                                      #tells us about DESCR key of diabetes dataset

diabetes_X=diabetes.data 

#print(diabetes_X)

diabetes_X_train=diabetes_X[:-30]
diabetes_X_test=diabetes_X[-30:]

diabetes_y_train=diabetes.target[:-30]
diabetes_y_test=diabetes.target[-30:]

model=linear_model.LinearRegression()
model.fit(diabetes_X_train,diabetes_y_train)

diabetes_y_predicted=model.predict(diabetes_X_test)

print("\nResults for MULTIPLE REGRESSION\n")
print("Mean squared error is: ", mean_squared_error(diabetes_y_test, diabetes_y_predicted))
print("Weights: ", model.coef_)
print("Intercept: ", model.intercept_, end="\n\n")
