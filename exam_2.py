import numpy as np
import pandas as pd

dataset = pd.read_csv(r"C:\Users\SJCET\Downloads\Company_data.csv")
print(dataset.head())
x=dataset.iloc[:,:-1]
y=dataset.iloc[:,1]

from sklearn.model_selection import  train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(regressor.intercept_)
print(regressor.coef_)

from  sklearn import metrics

print("The absolute error ",metrics.mean_absolute_error(y_test,y_pred))
print()
print("The mean square error ",metrics.mean_squared_error(y_test,y_pred))
print()
print("The Rot Mean square value ",np.sqrt(metrics.mean_squared_error(y_test,y_pred)))

