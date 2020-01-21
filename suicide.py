#!/usr/bin/env python
# coding: utf-8

# In[1]:
import pandas as pd
import numpy as np # linear algebra
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

suicide = pd.read_csv('master.csv',index_col=0,parse_dates=[0])

#Originally "Country" is the index column, this creates an index column
suicide = suicide.reset_index()


#we delete columns we dont need
del suicide['country-year']         #Redundant
del suicide['HDI for year']         #Dont know what it is
del suicide['suicides_no']          #Collapes into Suicides/pop
del suicide['population']
del suicide[' gdp_for_year ($) ']   #We keep gdp_per_capita

#We create a new categorical variable "Region" with all null values
header = ['country',
 'year',
 'sex',
 'age',
 'suicides/100k pop',
 'gdp_per_capita ($)',
 'generation',
 'region']

suicide = suicide.reindex(columns = header)        

#We manually put all the differnt countries into one of 6 regions
Europe = ["Albania","Russian Federation","France","Ukraine","Germany","Poland","United Kingdom",
         "Italy","Spain","Hungary","Romania","Belgium","Belarus","Netherlands","Austria",
         "Czech Republic","Sweden","Bulgaria","Finland","Lithuania","Switzerland","Serbia",
         "Portugal","Croatia","Norway","Denmark","Slovakia","Latvia","Greece","Slovenia",
         "Turkey","Estonia","Georgia","Albania","Luxembourg","Armenia","Iceland","Montenegro",
         "Cyprus","Bosnia and Herzegovina","San Marino","Malta","Ireland"]

NorthAmerica = ["United States","Mexico","Canada","Cuba","El Salvador","Puerto Rico",
                "Guatemala","Costa Rica","Nicaragua","Belize","Jamaica"]

SouthAmerica = ["Brazil","Colombia", "Chile","Ecuador","Uruguay","Paraguay","Argentina",
                "Panama","Guyana","Suriname"]

MiddleEast = ["Kazakhstan","Uzbekistan","Kyrgyzstan","Israel","Turkmenistan","Azerbaijan",
              "Kuwait","United Arab Emirates","Qatar","Bahrain","Oman"]

Asia = ["Japan","Republic of Korea", "Thailand", "Sri Lanka","Philippines","New Zealand",
        "Australia","Singapore","Macau","Mongolia"]

#if the country belongs to a region, we assign the observation with the region
for i in range(0,len(suicide)):
    if suicide.iloc[i,0] in Europe:
        suicide.iloc[i,7] = "Europe"
    elif suicide.iloc[i,0] in NorthAmerica:
        suicide.iloc[i,7] = "North America"
    elif suicide.iloc[i,0] in SouthAmerica:
        suicide.iloc[i,7] = "South America"
    elif suicide.iloc[i,0] in MiddleEast:
        suicide.iloc[i,7] = "Middle East"
    elif suicide.iloc[i,0] in Asia:
        suicide.iloc[i,7] = "Asia"
    else:
        suicide.iloc[i,7] = "Island Nation"

#Now that we dont need "country", we delete it. 
del suicide['country']

# #We collect our categorial variables for OneHotEncoding
suicide_cat = suicide[['sex','age','generation','region']]
# A dummy variable is a numerical variable used in regression analysis to represent subgroups of the sample in your study.
one_hot_data = pd.get_dummies(suicide_cat)

#We merge the data back together
year = suicide['year']
gdp_per_cap = suicide['gdp_per_capita ($)']
suicide_per_100k = suicide['suicides/100k pop']
data = pd.concat([year, gdp_per_cap, one_hot_data], axis=1)
print(data)



#We use a GridSearchCV to search for the best hyperparameters. In total we sampled 21000 different trees.
#I've truncated it for speed. 

#perform Descion Tree
def Decision_Tree(Test_Size,Random_State):
    #Train the models
    X_Train, X_Test, Y_Train, Y_Test = train_test_split(data, suicide_per_100k, test_size= Test_Size, random_state= Random_State)

    params = {'max_leaf_nodes': list(range(93,95)), 'min_samples_split': list(range(6,8)), 'min_samples_leaf':list(range(2,4))}    
    grid_search_cv = GridSearchCV(DecisionTreeRegressor(random_state=42),
                                params, n_jobs=-1, verbose=1, cv=3)

    grid_search_cv.fit(X_Train, Y_Train)

    y_pred = grid_search_cv.predict(X_Test)
    tree_reg_mse = mean_squared_error(Y_Test, y_pred)
    tree_reg_rmse = np.sqrt(tree_reg_mse)

    print("The Root-Mean-Squared Error for Decision Tree Regression model is :",tree_reg_rmse)

    #Return the Root-Mean-Squared-Error Value
    return tree_reg_rmse

#perform Random Forest Aggresor
def Random_Forest_Aggresor(Test_Size,Random_State):
    #Train the models
    X_Train, X_Test, Y_Train, Y_Test = train_test_split(data, suicide_per_100k, test_size= Test_Size, random_state= Random_State)

    rfg = RandomForestRegressor()
    rfg.fit(X_Train,Y_Train)

    y_pred_rfg = rfg.predict(X_Test)

    rfg_reg_mse = mean_squared_error(Y_Test, y_pred_rfg)
    
    rfg_reg_rmse = np.sqrt(rfg_reg_mse)
    print("The Root-Mean-Squared Error for Random Forest regression model is :",rfg_reg_rmse)
  
    #Return the Root-Mean-Squared-Error Value
    return rfg_reg_rmse

#perform Support Vector Regresor

def Support_Vector_Regresor(Test_Size,Random_State):
    #Train the models

    X_Train, X_Test, Y_Train, Y_Test = train_test_split(data, suicide_per_100k, test_size= Test_Size, random_state= Random_State)

    Lin_svr = SVR(C=1.0, epsilon=0.4)
    Lin_svr.fit(X_Train,Y_Train)

    y_pred_lin_svr = Lin_svr.predict(X_Test)

    svr_reg_mse = mean_squared_error(Y_Test, y_pred_lin_svr)
    # plt.plot(Y_Test)
    svr_reg_rmse = np.sqrt(svr_reg_mse)
    print("TThe Root-Mean-Squared Error for SVR regression model is :",svr_reg_rmse)

    #Return the Root-Mean-Squared-Error Value
    return svr_reg_rmse

#perfrom linear regression

def Linear_Regression(Test_Size,Random_State):
    #Train the models

    X_Train, X_Test, Y_Train, Y_Test = train_test_split(data, suicide_per_100k, test_size= Test_Size, random_state= Random_State)

    lin_reg = LinearRegression()
    lin_reg.fit(X_Train,Y_Train)

    y_pred_lin_reg = lin_reg.predict(X_Test)
    lin_reg_mse = mean_squared_error(Y_Test, y_pred_lin_reg)
    lin_reg_rmse = np.sqrt(lin_reg_mse)
    print("The Root-Mean-Squared Error for linear regression model is :",lin_reg_rmse)

    #Return the Root-Mean-Squared-Error Value
    return lin_reg_rmse


def Plot_RMSE(SVR,Linear,DT,RFA):
	names = ['SVR', 'Linear', 'D_Tree', 'RFA']
	values = [SVR, Linear, DT, RFA]
	plt.suptitle("Root Mean Squared Error")
	plt.xlabel("RMSE Comparison for Each Supervised Learning Method")
	plt.ylabel("Method Name")
	plt.barh(names, values, align='center', alpha=0.5)
	plt.yticks(names) 
	plt.show()

def main():

    DT = Decision_Tree(0.6 , 42)

    Linear = Linear_Regression(0.6, 42)
   
    SVR= Support_Vector_Regresor(0.6, 42)

    RFA = Random_Forest_Aggresor(0.6, 42)

    Plot_RMSE (SVR,Linear, DT, RFA)


main()

