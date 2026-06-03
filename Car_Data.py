import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn import metrics

warnings.filterwarnings("ignore")

data = pd.read_csv('car_data.csv')

print(data.head())
print(data.shape)
print(data.dtypes)


x = data.drop('price_usd' , axis=1)
y = data['price_usd']


#print(f" x : {x} \n y : {y}")


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures


x_train , x_test , y_train , y_test = train_test_split(x , y , test_size= 0.2 ,random_state=40)

regressor = LinearRegression()
print("Linear Regression :")
regressor.fit(x_train, y_train)
y_predict = regressor.predict(x_test)
print("R2 Score : " , r2_score(y_test, y_predict))
print("Mean Absolute Error :" , metrics.mean_absolute_error(y_test, y_predict))

print("Polynomial Regression :")
regressor_poly = LinearRegression()
poly_feature = PolynomialFeatures(degree = 3)
x_train_poly = poly_feature.fit_transform(x_train)
regressor_poly.fit(x_train_poly, y_train)
y_train_predict = regressor_poly.predict(x_train_poly)
y_test_predict = regressor_poly.predict(poly_feature.fit_transform(x_test))
print("R2 Score : " , r2_score(y_test,y_test_predict))