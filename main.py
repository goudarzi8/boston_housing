

#-----------------------------------------------------------------------------#
#-----------------------------------------------------------------------------#
#---------------------------This is the main App------------------------------#
#-----------------------------------------------------------------------------#
#-----------------------------------------------------------------------------#



#--------------------------------Libraries------------------------------------#

import pandas as pd 
#Importing pandas to use analytical tools 
#to handle tables and data

import numpy as np
#Numpy helps using more mathematical functions in python

import matplotlib.pyplot as pl
#Drawing graphs

import sklearn.metrics as metrics
#Calculating R Square and evaluating the model

from sklearn.cross_validation import train_test_split
#IMporting this to split the data and get them ready for testing

from sklearn.cross_validation import ShuffleSplit
#Using this library to apply cross validation

from sklearn.tree import DecisionTreeRegressor
#Importing Decision Tree Regression model

from sklearn.grid_search import GridSearchCV
#GridSearchCV implements a fit and a score method

#---------------------------------Reading Data----------------------------------#

def loading():

	data = pd.read_csv('data/data.csv') 
	# loading the data file

	prices = data['MEDV'] 
	# loading prices for easier calculations

	attributes = data.drop('MEDV', axis = 1) 
	# Removing MEDV from the data to control them separately
	return prices,attributes

#------------------------Spliting and Preparing Data----------------------------#

def spliting(attributes,prices,test=0.3,train=0.7):
	# Spliting the data into 7:3 for training and testing randomly

	return train_test_split(attributes,prices,test_size=test,train_size=train,random_state=7)


#------------------------R^2 Performance Calculation----------------------------#

def r2calc(y_true, y_predict):
	# Creating a function to do the R^2 evaluation
    
	r2calc_result = metrics.r2_score(y_true, y_predict)
    # Calculating R^2 using Scikit library

    
	return r2calc_result
    # Returning R^2

#------------------------ Error Performance Calculation----------------------------#

def meanSquaredError(y_true, y_predict):
	# Creating a function to do the mean squad error evaluation
    
	mse_result = metrics.mean_squared_error(y_true, y_predict)
    # Calculating mean squad error using Scikit library

    
	return mse_result
    # Returning mean squad error

#------------------------------Drawing Graph -----------------------------------#

def curveDraw(sizes, train_err, test_err,depth):

	# Drawing the training error vs the testing error to find the optimum depth

	pl.figure()

	pl.title ('Performance vs Training Size Using Depth = ' + str(depth))

	pl.plot(sizes, test_err, lw=2, color = 'Black',label = 'Test Error')

	pl.plot(sizes, train_err, lw=2, color = 'Blue', label = 'Training Error')

	pl.legend()

	pl.xlabel('Training Size')

	pl.ylabel('Error')

	pl.show()

#---------------------------DecisionTreeRegressor-------------------------------#

def decisionTreeReg(depth, X_train, y_train, X_test, y_test):
	# Decision Tree Regresison Training and evaluation function 

	sizes = np.linspace(1, len(X_train), 50)
	#creating up to 50 different training sizes
	train_err = np.zeros(len(sizes))

	test_err = np.zeros(len(sizes))

	for i, s in enumerate(sizes):

		regressor = DecisionTreeRegressor(max_depth=depth)
        # Create decision tree regression model

		regressor.fit(X_train[:int(s)], y_train[:int(s)])
        # Training the model

		train_err[i] = meanSquaredError(y_train[:int(s)], regressor.predict(X_train[:int(s)]))
        # Finding the performance on the training set

		test_err[i] = meanSquaredError(y_test, regressor.predict(X_test))
        # Finding the performance on the testing set

	curveDraw(sizes, train_err, test_err,depth)
    # drawing the graph

#--------------------------- Fitting Model-------------------------------#

def fit_model(X, y):
    
    cv_sets = ShuffleSplit(X.shape[0], n_iter = 15, test_size = 0.30, random_state = 0)
    # Making Cross-validation training data sets

    regressor = DecisionTreeRegressor()
    # Initiating decision tree regressor

    params = {"max_depth":range(1,10)}
    #Creating a dictionary of 1 to 10

    # TODO: Transform 'performance_metric' into a scoring function using 'make_scorer' 
    scoring_fnc = metrics.make_scorer(r2calc)

    # TODO: Create the grid search object
    grid = GridSearchCV(regressor, params, scoring_fnc, cv=cv_sets)

    # Fit the grid search object to the data to compute the optimal model
    grid = grid.fit(X, y)

    # Return the optimal model after fitting the data
    return grid.best_estimator_

#------------------------------ Main Function ----------------------------------#

def main():
#This is the main function

	print("Loading Data")
	prices,attributes = loading()
	#Loading data

	print("Spliting the Data")
	X_train, X_test,y_train,y_test = spliting(attributes,prices)
	#Spliting data into training and testing sets

	max_depths = [1,2,3,4,5,6,7,8,9,10]
	#Choosing different Max_Depth to find the optimum model
	
	for max_depth in max_depths:
		print ("Using Max_Depth = " + str(max_depth))
		decisionTreeReg(max_depth, X_train, y_train, X_test, y_test)
    # Decision Tree regression curves are created in this part

	reg = fit_model(X_train, y_train)
	# Fit the training data to the model using grid search

	print "Parameter 'max_depth' is {} for the optimal model.".format(reg.get_params()['max_depth'])
	# Produce the value for 'max_depth'

#------------------------------- Start System ----------------------------------#

if __name__ == '__main__':
	main()
	#This will direct user to the main function on the execution of the main file