import numpy as np
import pandas as pd
import math
import xlrd
from sklearn import linear_model

#Reading data from excel and rounding values on 2 decimal places
data = pd.read_excel("DataSet.xls").round(2)
data_size = data.shape[0]

def linear_regression(input_data, var1, var2, var3,
                      var4 , var5, var6, var7, var8):
    
    variables = input_data.iloc[:,:-1]
    results = input_data.iloc[:,-1]
    
    #creating linear model and fitting data into linear model
    regression = linear_model.LinearRegression()
    model = regression.fit(variables, results)

    input_values = [var1, var2, var3, var4 , var5, var6, var7, var8]

    #making the prediction based on inputed values
    prediction = regression.predict([input_values])
    prediction = round(prediction[0], 2)

    return "Linear prediction: " + str(prediction)

print(linear_regression(data, 260.9, 100.5, 78.3, 200.6, 8.6, 864.5, 761.5, 28)) #true value affter 28 days: 32.40 MPa
