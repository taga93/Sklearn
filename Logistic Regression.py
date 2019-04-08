import numpy as np
import pandas as pd
import math
import xlrd
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression

#Reading data from excel and rounding values on 2 decimal places
data = pd.read_excel("DataSet.xls").round(2)
data_size = data.shape[0]

#creating new dataframe with one additional column for concrete class
def concrete_strength_class(input_data):

    MB_list = []
    for row in input_data["MPa"]:

        if row < 35:
            MB_list.append("less than 35 MPa")

        else:
            MB_list.append("more than  35 MPa")

    data_frame_concrete_class = input_data.copy()
    data_frame_concrete_class["MB"] = MB_list

    return data_frame_concrete_class


def LogisticRegression(input_data, var1, var2, var3,
                        var4, var5, var6, var7, var8):

    data_for_classification = concrete_strength_class(input_data)
    
    variables = data_for_classification.iloc[:,:-2]
    results = data_for_classification.iloc[:,-1]

    #creating logistic model and fitting data into logistic regression model
    logistic_regression = linear_model.LogisticRegression(solver="lbfgs")
    model = logistic_regression.fit(variables, results)

    input_values = [var1, var2, var3, var4 , var5, var6, var7, var8]

    #making the prediction based on inputed values
    prediction = logistic_regression.predict([input_values]) #adding values for prediction
    prediction = prediction[0]
    
    return "Prediction with decision tree: " + str(prediction)

print(LogisticRegression(data, 260.9, 100.5, 78.3, 200.6, 8.6, 864.5, 761.5, 28)) #true value affter 28 days: 32.40 MPa

    
