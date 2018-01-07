

#-----------------------------------------------------------------------------#
#-----------------------------------------------------------------------------#
#---------------------------This is the main App------------------------------#
#-----------------------------------------------------------------------------#
#-----------------------------------------------------------------------------#



#--------------------------------Libraries------------------------------------#

import pandas as pd 
#Importing pandas to use analytical tools 
#to handle tables and data

import sklearn.metrics as metrics
# calculating R Square and evaluating the model

from sklearn.cross_validation import train_test_split
# IMporting this to split the data and get them ready for testing

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression

#---------------------------------Reading Data----------------------------------#

data = pd.read_csv('data/data.csv') 
# loading the data file

prices = data['MEDV'] 
# loading prices for easier calculations

attributes = data.drop('MEDV', axis = 1) 
# Removing MEDV from the data to control them separately

#------------------------Spliting and Preparing Data----------------------------#

attributes = StandardScaler().fit_transform(attributes)
# Fitting the data

X_train, X_test,y_train,y_test = train_test_split(attributes,prices,test_size=0.30,train_size=0.70,random_state=7)
# Spliting the data into 7:3 for training and testing randomly


#------------------------R^2 Performance Calculation----------------------------#
def r2calc(y_true, y_predict):
# Creating a function to do the R^2 
    
    r2calc_result = metrics.r2_score(y_true, y_predict)
    # Calculating R^2 using Scikit library
    
    return r2calc_result
    # Returning R^2

#-----------------------------Linear Regression---------------------------------#

def linearReg():
# Analysing linear regression

	clf = LinearRegression()
	clf.fit(X_train, y_train)
	#fitting training data
	test_result = clf.predict(X_test)
	#Testing data
	return r2calc(y_test,test_result)
	#Returning the performance of the linear regression 

#------------------------------ Main Function ----------------------------------#

def main():
	print("testing")

#------------------------------- Start System ----------------------------------#

if __name__ == '__main__':
    main()
#This will send the user to the main function on the execution of function