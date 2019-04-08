import numpy as np
import pandas as pd
import math
import xlrd
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures

#Reading data from excel and rounding values on 2 decimal places
data = pd.read_excel("DataSet.xls").round(2)
data_size = data.shape[0]

def polynomial_regression(input_data, var1, var2, var3,
                          var4, var5, var6, var7, var8):

    variables = input_data.iloc[:,:-1]
    results = input_data.iloc[:,-1]
    
    #reshaping the values so that
    #'variables' and 'results' have the same shape
    n = results.shape[0]
    results = results.values.reshape(n, 1)

    #transforming data into polynomial function if 2nd degree
    PolynomialFunction = PolynomialFeatures(degree=2)
    polynomial_variables = PolynomialFunction.fit_transform(variables)

    #creating linear model and fitting data into linear model
    regression = linear_model.LinearRegression()
    model = regression.fit(polynomial_variables, results)

    #transforming input data for prediction into polynomial function
    input_values = [var1, var2, var3, var4 , var5, var6, var7, var8]
    input_values = PolynomialFunction.transform([input_values])
    
    #making the prediction based on inputed values
    prediction = regression.predict(input_values)
    prediction = round(prediction[0,0], 2)

    return "Polynomial prediction: " + str(prediction)

print(polynomial_regression(data, 260.9, 100.5, 78.3, 200.6, 8.6, 864.5, 761.5, 28)) #true value affter 28 days: 32.40 MPa
