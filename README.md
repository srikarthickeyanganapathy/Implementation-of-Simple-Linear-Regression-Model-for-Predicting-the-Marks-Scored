# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import pandas, numpy and sklearn
2. Calculate the values for the training data set
3. Calculate the values for the test data set
4. Plot the graph for both the data sets and calculate for MAE, MSE and RMSE

## Program:
```
Program to implement the linear regression using gradient descent.
Developed by: SRI KARTHICKEYAN GANAPATHY
RegisterNumber: 212222240102

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
df=pd.read_csv('/student_scores.csv')
#Displying the contents in datafile
df.head()

df.tail()

#Segregating data to variables
X=df.iloc[:,:-1].values
X

Y=df.iloc[:,-1].values
Y

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)

#displaying the predicted values
Y_pred

#displaying the actual values
Y_test

#graph plot for training data
plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="green")
plt.title("Hours vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

#graph plot for test data
plt.scatter(X_test,Y_test,color="blue")
plt.plot(X_test,regressor.predict(X_test),color="black")
plt.title("Hours vs Scores(Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(Y_test,Y_pred)
print('MSE= ',mse)
mae=mean_absolute_error(Y_test,Y_pred)
print("MAE= ",mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```

## Output:
### df.head():
![mll 1](https://github.com/srikarthickeyanganapathy/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393842/7eeecb9e-dda5-4745-9a14-5e9f31f7dabc)

### df.tail():
![mll 2](https://github.com/srikarthickeyanganapathy/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393842/9ce25f08-9237-4cab-b66d-1de58c4e8326)

### X:
![mll 3](https://github.com/srikarthickeyanganapathy/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393842/e3b7f790-dc18-40b3-97d6-fac227f58fc5)

### Y:
![mll 4](https://github.com/srikarthickeyanganapathy/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393842/6ad2b1ce-1737-4707-936e-51e63bd171ae)

### PREDICTED Y VALUES:
![mll 5](https://github.com/srikarthickeyanganapathy/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393842/289a03ec-1dbc-4193-bbfc-3b126ac4af4d)

### ACTUAL Y VALUES:
![mll 6](https://github.com/srikarthickeyanganapathy/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393842/1fbe8c04-fc66-4f0a-b598-532d2e4f993e)

### GRAPH FOR TRAINING DATA:
![mll 7](https://github.com/srikarthickeyanganapathy/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393842/4bd6cd90-3db7-4ffc-93bf-a39dc4fd6504)

### GRAPH FOR TEST DATA:
![mll 8](https://github.com/srikarthickeyanganapathy/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393842/7abbff2a-4d3f-48e5-bc38-ff6156cd853d)

### MEAN SQUARE ERROR, MEAN ABSOLUTE ERROR AND RMSE:
![mll 9](https://github.com/srikarthickeyanganapathy/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393842/f0123eb6-bbc1-4b3e-aad8-4ff724e44bca)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
