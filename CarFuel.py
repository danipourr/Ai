import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv('Car_Fuel.csv')

print(data.head())
print(data.shape)
print(data.dtypes)


x = data[['Speed_KMH']]
y = data[['Fuel_Consumption']]

#print(f"X = {x} \n Y = {y}")


from sklearn.model_selection import train_test_split

x_train , x_test , y_train , y_test = train_test_split(x , y , test_size = 0.2 , random_state = 42)

# print(x_train.shape)
# print(y_train.shape)
# print(x_test.shape)
# print(y_test.shape)


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics

poly_feature = PolynomialFeatures(degree = 2)
x_train_poly = poly_feature.fit_transform(x_train)
x_test_poly = poly_feature.transform(x_test)


regressor = LinearRegression()
regressor.fit(x_train_poly,y_train)

y_train_poly = regressor.predict(x_train_poly)
y_test_poly = regressor.predict(x_test_poly)

print("R^2 Score Train :" , metrics.r2_score(y_train , y_train_poly))
print("R^2 Score Test  : " , metrics.r2_score(y_test , y_test_poly))

