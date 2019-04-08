import numpy as np
import pandas as pd
import math
import xlrd
from sklearn import tree

#Reading data from excel and rounding values on 2 decimal places
data = pd.read_excel("DataSet.xls").round(2)
data_size = data.shape[0]

#creating new dataframe with one additional column for concrete class
def concrete_strength_class(input_data):

    MB_list = []
    for row in input_data["MPa"]:

        if row < 10:
            MB_list.append("nan/<10 MPa")
            
        elif row >= 10 and row <= 20:
            MB_list.append("MB 10-15")

        elif row > 20 and row <= 30:
            MB_list.append("MB 20-25")
            
        elif row > 30 and row <= 40:
            MB_list.append("MB 30-35")
            
        elif row > 40 and row <= 50:
            MB_list.append("MB 40-45")
            
        elif row > 50 and row <=60:
            MB_list.append("MB 50-55")

        else:
            MB_list.append(">MB 60")

    data_frame_concrete_class = input_data.copy()
    data_frame_concrete_class["MB"] = MB_list

    return data_frame_concrete_class


def decision_tree_classifier(input_data, var1, var2, var3,
                             var4, var5, var6, var7, var8):

    data_for_classification = concrete_strength_class(input_data)
    
    variables = data_for_classification.iloc[:,:-2]
    results = data_for_classification.iloc[:,-1]

    #creating decision tree model and fitting data into decision tree model
    decision_tree = tree.DecisionTreeClassifier()
    model = decision_tree.fit(variables, results)

    input_values = [var1, var2, var3, var4, var5, var6, var7, var8]
    
    #making the prediction based on inputed values
    prediction = decision_tree.predict([input_values])
    prediction = prediction[0]

    return "Prediction with decision tree: " + str(prediction)
  

print(decision_tree_classifier(data, 260.9, 100.5, 78.3, 200.6, 8.6, 864.5, 761.5, 28)) #true value affter 28 days: 32.40 MPa


