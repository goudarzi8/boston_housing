import pandas as pd 
#Importing pandas to use analytical tools 

from sklearn.datasets import *
from sklearn.linear_model import LinearRegression
# Applying simple regression analysis


data = pd.read_csv('data/data.csv') 
# loading the data file

print("MIN Value for each Attribute is")
print(data.min())

print("First, Seccond and Third Quantile Value for each Attribute is")
print(data.quantile([0.25]))

print("Median Value for each Attribute is")
print(data.median())

print("Third Quantile Value for each Attribute is")
print(data.quantile([0.75]))

print("Max Value for each Attribute is")
print(data.max())

print("Average Value for each Attribute is")
print(data.mean())

print("Standard Deviation Value for each Attribute is")
print(data.std())





prices = data['MEDV'] 
# loading prices for easier calculations

attributes = data.drop('MEDV', axis = 1) 
# Removing MEDV from the data to control them separately

model = LinearRegression()
model.fit(attributes, prices)
print model.__dict__
print model.score(attributes,prices)
