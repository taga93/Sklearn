import numpy as np
import pandas as pd
import math
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import FunctionTransformer

#Reading data from excel and rounding values on 2 decimal places
data = pd.read_excel("DataSet.xls").round(2)
data_size = data.shape[0]

#some values are 0, so I need to eliminate them because I cant do log 0 function
my_data = data[(data["Superpl"] == 0) &
               (data["FlyAsh"] == 0) &
               (data["BlastFurSlag"] == 0)].drop(columns=["Superpl","FlyAsh","BlastFurSlag"])


def logarithmic_regression(input_data, cement, water, coarse_aggr, fine_aggr, days):

    variables = input_data.iloc[:,:-1]
    results = input_data.iloc[:,-1]
    
    n = results.shape[0]
    results = results.values.reshape(n,1) #reshaping the values so that variables and results have the same shape

    #transforming x data to logarithmic fucntion
    log_regression = FunctionTransformer(np.log, validate=True)
    log_variables = log_regression.fit_transform(variables)

    #making linear model and fitting the logarithmic data into linear model
    regression = linear_model.LinearRegression() 
    model = regression.fit(log_variables, results)
        
    input_values = [cement, water, coarse_aggr, fine_aggr, days]
        
    #transforming input data for prediction in logarithmic function
    input_values = log_regression.transform([input_values]) 

    #predicting the outcome based on the input_values
    predicted_strength = regression.predict(input_values) #adding values for prediction
    predicted_strength = round(predicted_strength[0,0], 2)

    return "Logarithmic prediction: " + str(predicted_strength)


print(logarithmic_regression(my_data, 339.0, 197.0, 968.0, 781.0, 14))
