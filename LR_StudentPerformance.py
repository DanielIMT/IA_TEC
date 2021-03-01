"""load DS into a data frame"""
import pandas as pd
import numpy as np
df = pd.read_csv('student-por.csv', usecols= ["sex","age","address","famsize", "Pstatus", "Medu", "Fedu", "traveltime", "studytime", "activities", "internet", "romantic", "health", "absences", "G3"])
df.fillna(0,inplace=True)   #Fill with cero in case of empty data
print(df)
features = ["sex","age","address","famsize", "Pstatus", "Medu", "Fedu", "traveltime", "studytime", "activities", "internet", "romantic", "health", "absences"]
df_X = df[features]
df_Y = df["G3"]



"""SPLIT IN TRAINING AND TEST DATASETS"""
from sklearn.model_selection import train_test_split
dfX_train, dfX_test, dfY_train, dfY_test = train_test_split(df_X,df_Y,test_size=0.2)#means 20 percent of DS goes to testing


"""SCALE DATA 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
dfX_train[:, [1,13]] = scaler.fit_transform(dfX_train[:, [1,13]])
dfX_test[:, [1,13]] = scaler.fit_transform(dfX_test[:, [1,13]])
"""



""""BUILD THE LINEAR REGRESION MODEL"""
from sklearn import linear_model
model = linear_model.LinearRegression()
model.fit(dfX_train, dfY_train)  #Using 80 percent of data for training
Y_pred = model.predict(dfX_test)

"""Prediction results"""
from sklearn.metrics import mean_squared_error, r2_score
print("\n---------------MODEL-------------------------")
print("Coefficients: {} \n".format(model.coef_))
print("Intercept: {}".format(model.intercept_))
print("Mean Squared error: {}".format(mean_squared_error(dfY_test, Y_pred)))
print("Coefficient of determination: {}".format(r2_score(dfY_test, Y_pred)))
print("---------------------------------------------\n")



"""USER INSTANCE"""
while(True):
    userData = []
    userData.append(int(input("What is your sex?--> 1.F   2.M")))
    userData.append(int(input("How old are u? --->")))
    userData.append(int(input("In which area do you live? --> 1.Urban   2.Rural")))
    userData.append(int(input("How many members are in your family --> 1. Less than four   2.Four or more")))
    userData.append(int(input("What is your parents status --> 1.Together   2.Apart")))
    userData.append(int(input("Mother's education--> 0.None  1.Primary  2.Secundary  3.Highschool  4.Higher education ")))
    userData.append(int(input("Father's education--> 0.None  1.Primary  2.Secundary  3.Highschool  4.Higher education ")))
    userData.append(int(input("Home to school travel time? --> 1. <15 min   2. 15 to 30 min   3. 30 min to 1 h   4.More than 1 h")))
    userData.append(int(input("How much do u study per week? --> 1. <2Hrs   2. 2 to 5 hrs   3. 5 to 10 hrs   4.More")))
    userData.append(int(input("Do u have extra curricular activities? --> 1. Yes  2.No")))
    userData.append(int(input("Do u have internet? --> 1.Yes   2.No")))
    userData.append(int(input("Are u in a relationship? --> 1.Yes  2.No")))
    userData.append(int(input("From very bad (1) to really good (5) how you consider your health?--> ")))
    userData.append(int(input("How many absences you have?-->")))

    df_user = pd.DataFrame([userData],columns = features)
    print("Your expected grade from 0 to 20 would be: {}".format(model.predict(df_user)))

    option= int(input("Again? --> 1.Yes  2.No"))
    if (option == 2):
        break

