import numpy as np
import pandas as pd
import matplotlib as mpl
# mpl.use('TkAgg)    for MAC pc only
import matplotlib.pyplot as plt

#import the dataSet
dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1:].values

# splitting the dataset  into Training and Test dataset
from sklearn.model_selection import train_test_split
# used model_selection in place of cross_validation since the latter is deprecated
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 1/3, random_state = 0)

#fitting simple linear regression to the training set
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train,y_train)

# predicting the Test set results
y_pred = regressor.predict(x_test)

#visualising the training results
plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,regressor.predict(x_train), color='blue')
plt.title('Salary Vs Experience (Training set)')
plt.xlabel('Years Of Experience')
plt.ylabel('Salary')
plt.show()

#visualising the test set results
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train), color='blue')
plt.title('Salary Vs Experience (Test set)')
plt.xlabel('Years Of Experience')
plt.ylabel('Salary')
plt.show()