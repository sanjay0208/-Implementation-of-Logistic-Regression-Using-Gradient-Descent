# Implementation-of-Logistic-Regression-Using-Gradient-Descent->

## AIM:

To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:

1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Use the standard libraries in python for finding linear regression.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Predict the values of array.
5. Calculate the accuracy, confusion and classification report b importing the required modules from sklearn.
6. Obtain the graph.

## Program:

```

Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: sanjay m
RegisterNumber:  212222110038


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data = np.loadtxt("ex2data1.txt",delimiter=",")
X = data[:,[0,1]]
Y = data[:,2]

X[:5]

Y[:5]

# VISUALIZING THE DATA
plt.figure()
plt.scatter(X[Y== 1][:, 0], X[Y==1][:,1],label="Admitted")
plt.scatter(X[Y==0][:,0],X[Y==0][:,1],label="Not admitted")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
plt.show()

def sigmoid(z):
    return 1/(1+np.exp(-z))

plt.plot()
X_plot=np.linspace(-10,10,100)
plt.plot(X_plot,sigmoid(X_plot))
plt.show()

def costFunction(theta, X, Y):
    h = sigmoid(np.dot(X, theta))
    J = -(np.dot(Y, np.log(h)) + np.dot(1-Y,np.log(1-h))) / X.shape[0]
    grad = np.dot(X.T, h-Y)/X.shape[0]
    return J,grad

X_train = np.hstack((np.ones((X.shape[0],1)),X))
theta = np.array([0,0,0])
J,grad = costFunction(theta,X_train,Y)
print(J)
print(grad)

X_train = np.hstack((np.ones((X.shape[0],1)),X))
theta = np.array([-24,0.2,0.2])
J,grad = costFunction(theta,X_train,Y)
print(J)
print(grad)

def cost(theta,X,Y):
  h=sigmoid(np.dot(X,theta))
  J=-(np.dot(Y,np.log(h))+np.dot(1-Y,np.log(1-h)))/X.shape[0]
  return J

def gradient(theta,X,Y):
  h=sigmoid(np.dot(X,theta))
  grad=np.dot(X.T,h-Y)/X.shape[0]
  return grad

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
res=optimize.minimize(fun=cost,x0=theta,args=(X_train,Y),method='Newton-CG',jac=gradient)
print(res.fun)
print(res.x)

def plotDecisionBoundary(theta,X,Y):
    X_min , X_max = X[:, 0].min() - 1,X[:,0].max() + 1
    Y_min , Y_max = X[:, 1].min() - 1,X[:,1].max() + 1
    XX,YY = np.meshgrid(np.arange(X_min,X_max,0.1),
                        np.arange(Y_min,Y_max,0.1))
    X_plot = np.c_[XX.ravel(), YY.ravel()]
    X_plot = np.hsatck((np.ones((X_plot.shape[0],1)),X_plot))
    Y_plot = np.dot(X_plot, theta).reshape(XX.shape)
    plt.figure()
    plt.scatter(X[Y==1][:,0],X[Y==1][:,1],label='Admitted')
    plt.scatter(X[Y==1][:,0],X[Y==1][:,1],label='Not admitted')
    plt.contour(XX,YY,Y_plot,levels=[0])
    plt.Xlabel("Exam 1 score")
    plt.Ylabel("Exam 2 score")
    plt.legend()
    plt.show()

print("Decision boundary-graph for exam score:")
plotDecisionBoundary(res.x,X,Y)


prob=sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)

def predict(theta, X):
  X_train=np.hstack((np.ones((X.shape[0],1)),X))
  prob=sigmoid(np.dot(X_train,theta))
  return (prob >= 0.5).astype(int)

np.mean(predict(res.x,X)==y)

```

## Output:

### Array value of X:

![image](https://github.com/Pradeeppachiyappan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118707347/64e299e3-52f6-476d-b33a-67e5bc6747b1)

### Array value of Y:

![image](https://github.com/Pradeeppachiyappan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118707347/d3015b94-8f3e-4498-8658-38437e78c8ee)

### Exam 1-Score graph:

![image](https://github.com/Pradeeppachiyappan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118707347/ebbef2ac-1dc6-4350-a007-a2bcb517ccc2)

### Sigmoid function graph:

![image](https://github.com/Pradeeppachiyappan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118707347/1e2b7e6f-981c-498b-811d-d16107d19c2e)

### X_Train_grad value:

![image](https://github.com/Pradeeppachiyappan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118707347/558e3e19-2b36-4caa-b41a-2e39f77c1ae3)

### Y_Train_grad value:

![image](https://github.com/Pradeeppachiyappan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118707347/1359be09-0139-4c55-a1b6-ebcb2f8a980f)

### Print res.X:

![image](https://github.com/Pradeeppachiyappan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118707347/bc898b21-2b10-4561-8364-f01df4971cf0)

### Decision boundary-gragh for exam score:

![image](https://github.com/Pradeeppachiyappan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118707347/b6e1d12e-3f12-442d-a433-d41a06a18cc0)

### Probability value:

![image](https://github.com/Pradeeppachiyappan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118707347/9b343fe5-1eaa-4428-a91f-01e9bcb8b9cc)

### Prediction value of mean:

![image](https://github.com/Pradeeppachiyappan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118707347/9da6578c-589c-4264-82db-fa4401958b39)

## Result:

Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.
