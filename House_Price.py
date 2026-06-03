import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


data = pd.read_csv('House_Price.csv')

#print(data.head())

data.drop(['Id'] , axis=1 , inplace=True)
print(data.head())

print(data.shape)
print(data.dtypes)
data['Price'] = data['Price'].str.replace(',', '', regex=False)
data['Price'] = pd.to_numeric(data['Price'])
print(data.isnull().sum())

x = data.drop(['Price'], axis=1)
y = data['Price']

#print(f"X = {x} , Y = {y}")

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

x_train , x_test , y_train , y_test = train_test_split(x , y , test_size = 0.2 , random_state = 42)

regressor = LinearRegression()
poly_feature = PolynomialFeatures(degree = 3)
x_train_poly = poly_feature.fit_transform(x_train)
x_test_poly = poly_feature.fit_transform(x_test)
regressor.fit(x_train_poly,y_train)
y_train_pred = regressor.predict(poly_feature.fit_transform(x_train))
y_test_pred = regressor.predict(poly_feature.fit_transform(x_test))
print(f"R2 Score : " , r2_score(y_train,y_train_pred))