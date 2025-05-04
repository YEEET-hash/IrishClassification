from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd


iris=load_iris()

x=pd.DataFrame(data=iris['data'],columns=iris['feature_names'])

y=pd.DataFrame(data=iris['target'],columns=['species'])

train_x,val_x,train_y,val_y=train_test_split(x,y)

model=DecisionTreeClassifier()

model.fit(train_x,train_y)

print(model.score(x,y))

