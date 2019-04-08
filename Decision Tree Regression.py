import numpy as np
import pandas as pd
import math
import xlrd
from sklearn.tree import DecisionTreeRegressor

#Reading data from excel and rounding values on 2 decimal places
data = pd.read_excel("DataSet.xls").round(2)
data_size = data.shape[0]

def decision_tree_regressor(input_data, var1, var2, var3,
                            var4 , var5, var6, var7, var8):

    variables = input_data.iloc[:,:-1]
    results = input_data.iloc[:,-1]
    
    #creating linear model and fitting data into linear model
    regression = DecisionTreeRegressor()
    model = regression.fit(variables, results)

    input_values = [var1, var2, var3, var4 , var5, var6, var7, var8]

    #making the prediction based on inputed values
    prediction = regression.predict([input_values])
    prediction = round(prediction[0], 2)

    return "Decision tree regressor prediction: " + str(prediction)

print(decision_tree_regressor(data, 500, 0, 0, 200, 0, 1125, 613, 3)) #true value affter 3 days: 26.02 MPa
print(decision_tree_regressor(data, 214.9 , 53.8, 121.9, 155.6, 9.6, 1014.3, 780.6, 7)) #true value affter 7 days: 18.20 MPa

