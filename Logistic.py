from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import numpy as np

def lossFunction(model,x,y):
    return np.log(1+np.exp(-y * model.predict(x)))

def empiricalErrorWithLossFunction(X,y_pred,model):
    sum = 0
    for i in range(X.shape[0]):
        sum += lossFunction(model,X[i].reshape(1,-1),y_pred[i])
    return sum / X.shape[0]

dataset = load_iris()
X = dataset.data
y = dataset.target

model = LogisticRegression(C=100.0)
model.fit(X,y)
y_pred = model.predict(X)

print(lossFunction(model,X,y))
print(empiricalErrorWithLossFunction(X,y_pred,model)) # error logistic regression