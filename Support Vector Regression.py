import pandas as pd
from sklearn import metrics
from sklearn.svm import SVR

#Reading data from excel
data = pd.read_excel("DataSet.xls").round(2)
data_size = data.shape[0]

def SupportVectorRegression(input_data, var1, var2, var3,
                            var4, var5, var6, var7, var8):

    variables = input_data.iloc[:,:-1]
    results = input_data.iloc[:,-1]
    
    #creating SVR model and fitting data into SVR model
    support_vector_regression = SVR(gamma = 'auto')
    model = support_vector_regression.fit(variables, results)

    input_values = [var1, var2, var3, var4, var5, var6, var7, var8]
    
    #making the prediction based on inputed values
    prediction = support_vector_regression.predict([input_values])
    prediction = round(prediction[0], 2)

    return "SVR prediction: " + str(prediction)

print(SupportVectorRegression(data, 260.9, 100.5, 78.3, 200.6, 8.6, 864.5, 761.5, 28))
