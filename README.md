# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

# AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

# Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

# Algorithm
1. Import the standard Libraries.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Assign the points for representing in the graph.
5. Predict the regression for marks by using the representation of the graph.
6. Compare the graphs and hence we obtained the linear regression for the given datas.

# Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Delli Priya L
RegisterNumber: 212222230029
*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('/student_scores.csv')
df.head()
df.tail()

## segregating data to variables
X=df.iloc[:,:-1].values
X
Y=df.iloc[:,1].values
Y

## graph plotting for training data
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)

## Dislaying predicted values
Y_pred

## Displaying actual values
Y_test

## Graph plot for training data
plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="blue")
plt.title("Hours Vs Scores(Train Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

## Graph plot for test data
plt.scatter(X_test,Y_test,color="purple")
plt.plot(X_test,regressor.predict(X_test),color="yellow")
plt.title("Hours vs scores (test set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE = ",rmse)

```

# Output:

## df.head()
![image](https://github.com/Priya-Loganathan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121166075/5878ff4a-0007-491d-bbf6-e67a56459c26)

## df.tail()
![image](https://github.com/Priya-Loganathan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121166075/7c57d20b-599e-48e0-948b-9f5dc38f6683)

## Array values of X
![image](https://github.com/Priya-Loganathan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121166075/9b7d8788-0c9e-4dec-8bf4-272abef7fc10)

## Array values of Y
![image](https://github.com/Priya-Loganathan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121166075/8a519420-4b47-4bd4-abf5-2a6abd5df0cd)

## Values of Y prediction
![image](https://github.com/Priya-Loganathan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121166075/d03caa1a-5751-4e51-9d63-b5b098a3f3ed)

## Values of Y test
![image](https://github.com/Priya-Loganathan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121166075/50ed55ee-5800-4d14-9c48-f632dc6837ed)

## Training set graph
![image](https://github.com/Priya-Loganathan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121166075/8b6e5ebb-2aa4-4707-be66-c2ff3bdd6120)

## Testing set graph
![image](https://github.com/Priya-Loganathan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121166075/870e9b84-7c65-43e5-850d-5e54772c205e)

## Value of MSE,MAE & RMSE
![image](https://github.com/Priya-Loganathan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121166075/97f8baf0-e0aa-4234-a520-8108c359b7cb)

# Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
