#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import xarray as xr
import netCDF4
import os
import datetime
import matplotlib.pyplot as plt 
from matplotlib import cm
import shutil
from datetime import datetime, timedelta
import glob
import datetime as dt
from os import path
import fsspec

from datetime import timezone
from sklearn.inspection import partial_dependence, PartialDependenceDisplay


# check scikit-learn version
import sklearn
from numpy import mean
from numpy import std
from sklearn import svm
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.utils import shuffle
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import pandas as pd
import glob
import os
from sklearn import preprocessing

from numpy import asarray
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import csv

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping


# In[ ]:


# In[ ]:


#from lazypredict.Supervised import LazyRegressor

from sklearn.utils import shuffle

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel


# In[ ]:


# Function to randomly sample 1 out of every 4 entries in each group
def sample_group(group):
    return group.sample(1)



# In[ ]:


import os, sys

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


# In[ ]:


def plot_importance(mean_importances,title,plot_folder):
    df = pd.DataFrame(mean_importances.sort_values(by=['Mean Importance'], ascending=False)["Mean Importance"].values)
    df.set_index(mean_importances.sort_values(by=['Mean Importance'], ascending=False).Feature.values,inplace=True)
    #plt.figure(figsize=(10, 5), dpi= 300)
    df.plot(kind='bar', stacked=True,legend=False,title=title, ylabel="relative importance")
    plt.tight_layout()
    plt.savefig(plot_folder+title+".pdf")
    plt.savefig(plot_folder+title+".png")
    plt.show()
    return


# In[ ]:


def compute_partial_dependence(model, X_train, feature_index, grid_values):
    # Create a list to store the predicted values for each grid point
    predicted_values = []
    
    # Iterate over each grid value
    for value in grid_values:
        # Make a copy of the dataset
        X_copy = X_train.copy()
        
        # Set the feature of interest (feature_index) to the grid value
        X_copy.iloc[:, feature_index] = value
        
        # Get predictions for the modified dataset
        predictions = model.predict(X_copy)
        
        # Average the predictions for this grid value
        predicted_values.append(np.mean(predictions))
    
    return predicted_values


# In[ ]:





# In[ ]:


def plot_feature_direction(data,var_choice,model_list,plot_folder,name_modifier="model",grid_points=50):
    X_train=data
    for feature in var_choice:
        feature_index = feature # which feature should be investigated
        min_value = X_train.iloc[:, feature_index].min()
        max_value = X_train.iloc[:, feature_index].max()
        grid_values = np.linspace(min_value, max_value, grid_points)  # 100 grid points
        
        # Function to compute partial dependence manually
        predicted_values_list=[]
        plt.figure(figsize=(10, 5), dpi= 300)
        
        for model in model_list:
        # Compute partial dependence manually
            predicted_values_A = compute_partial_dependence(model, X_train, feature_index, grid_values)
            predicted_values_list.append(predicted_values_A)
        
            
            
            # Plot the results
            #
            
            plt.plot(grid_values, predicted_values_A,color="black",alpha=0.02)
        
            
            
            # Print the results
            #print("Grid Values:", grid_values)
            #print("Predicted Values:", predicted_values)
        predicted_values_list_mean=np.mean(predicted_values_list,axis=0)
        predicted_values_list_med=np.median(predicted_values_list,axis=0)
        predicted_values_list_quant01=np.quantile(predicted_values_list,q=0.1,axis=0)
        predicted_values_list_quant09=np.quantile(predicted_values_list,q=0.9,axis=0)
        plt.plot(grid_values, predicted_values_list_mean,color="red",alpha=1,linewidth=3,label="mean")
        plt.plot(grid_values, predicted_values_list_med,color="blue",alpha=1,linewidth=3,label="median")
        plt.plot(grid_values, predicted_values_list_quant01,color="purple",alpha=1,linewidth=2,label="0.1 Quantile")
        plt.plot(grid_values, predicted_values_list_quant09,color="purple",alpha=1,linewidth=2,label="0.9 Quantile")
        plt.xlabel(X_train.columns[feature_index]+' Values')
        plt.ylabel('Predicted Values')
        title='Partial Dependence Plot for '+X_train.columns[feature_index]
        plt.title(title)
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_folder+title+"_"+name_modifier+".pdf")
        plt.savefig(plot_folder+title+"_"+name_modifier+".png")
        plt.show()
        plt.figure(figsize=(10, 5), dpi= 300)
        plt.plot(grid_values, predicted_values_list_mean,color="red",alpha=1,linewidth=3,label="mean")
        plt.plot(grid_values, predicted_values_list_med,color="blue",alpha=1,linewidth=3,label="median")
        plt.xlabel(X_train.columns[feature_index]+' Values')
        plt.ylabel('Predicted Values') 
        plt.title(title)
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_folder+title+"_"+name_modifier+"_mm.pdf")
        plt.savefig(plot_folder+title+"_"+name_modifier+"_mm.png")
        plt.show()
    return print("plots are saved in "+plot_folder)


# In[ ]:



# In[ ]:


def run_two_rf_BS(data,group_by,target,vars_2_rm_A,vars_2_rm_B,trainvalsplit=.2):
    data_sampled = data.groupby(group_by).apply(sample_group).reset_index(drop=True)
    data_A=data.drop(columns=vars_2_rm_A)
    data_B=data.drop(columns=vars_2_rm_B)
    
    data_features=data_sampled.drop(columns=vars_2_rm_A)
    feature_list=data_features.columns
    data_labels=data_sampled[target]
    data_features_shuffled,data_labels_shuffled=shuffle(data_features,data_labels,)
    X = data_features_shuffled.values
    y= data_labels_shuffled.values

    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=trainvalsplit)
    Z_train=np.repeat(1,np.shape(y_train)[0])
    Z_train=Z_train.reshape(np.shape(y_train)[0],1)
    Z_test=np.repeat(1,np.shape(y_test)[0])
    Z_test=Z_test.reshape(np.shape(y_test)[0],1)

    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=trainvalsplit)
    Z_train=np.repeat(1,np.shape(y_train)[0])
    Z_train=Z_train.reshape(np.shape(y_train)[0],1)
    Z_test=np.repeat(1,np.shape(y_test)[0])
    Z_test=Z_test.reshape(np.shape(y_test)[0],1)
    X_train_pd=data_features_shuffled.iloc[:np.shape(X_train)[0],:]
    y_train_pd=data_labels_shuffled.iloc[:np.shape(X_train)[0]]
    X_test_pd=data_features_shuffled.iloc[np.shape(X_train)[0]:,:]
    y_test_pd=data_labels_shuffled.iloc[np.shape(X_train)[0]:]


    # normal RF # normal RF
    model_A=RandomForestClassifier(n_estimators=500)
    model_A.fit(X_train_pd,y_train_pd)
    #model_A_explainer=shap.TreeExplainer(model_A)
    #model_A_shap_values=model_A_explainer.shap_values(X_train)
    pred_train_A = model_A.predict(X_train_pd)
    pred_test_A = model_A.predict(X_test_pd)
    acc_A=accuracy_score(pred_test_A, y_test_pd)
    prec_A=precision_score(pred_test_A, y_test_pd)
    rec_A=recall_score(pred_test_A, y_test_pd)
    importances = list(model_A.feature_importances_)
    feature_importances = [(feature, round(importance, 4)) for feature, importance in zip(feature_list, importances)]
    feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
    df_A = pd.DataFrame(feature_importances,columns=["feature","value"])
    df_A = pd.DataFrame(df_A.value.values, index=df_A.feature, columns=['Importance'])

    data_features=data_sampled.drop(columns=vars_2_rm_B)
    feature_list=data_features.columns
    data_labels=data_sampled[target]
    data_features_shuffled,data_labels_shuffled=shuffle(data_features,data_labels,)
    X = data_features_shuffled.values
    y= data_labels_shuffled.values

    # normal RF # normal RF
    model_B=RandomForestClassifier(n_estimators=500)
    model_B.fit(X_train_pd,y_train_pd)
    #model_B_explainer=shap.TreeExplainer(model_B)
    #model_B_shap_values=model_B_explainer.shap_values(X_train)
    importances = list(model_B.feature_importances_)
    pred_train_B = model_B.predict(X_train_pd)
    pred_test_B = model_B.predict(X_test_pd)
    acc_B=accuracy_score(pred_test_B, y_test_pd)
    prec_B=precision_score(pred_test_B, y_test_pd)
    rec_B=recall_score(pred_test_B, y_test_pd)
    feature_importances = [(feature, round(importance, 4)) for feature, importance in zip(feature_list, importances)]
    feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
    df_B = pd.DataFrame(feature_importances,columns=["feature","value"])
    df_B = pd.DataFrame(df_B.value.values, index=df_B.feature, columns=['Importance'])

    return df_A,df_B,acc_A,acc_B,prec_A,prec_B,rec_A,rec_B,model_A,model_B,data_A,data_B




def plot_feature_direction_grayspace(data, var_choice, model_list, plot_folder, name_modifier="model", grid_points=50):
    X_train = data
    for feature in var_choice:
        feature_index = feature  # Which feature to investigate
        min_value = X_train.iloc[:, feature_index].min()
        max_value = X_train.iloc[:, feature_index].max()
        grid_values = np.linspace(min_value, max_value, grid_points)

        predicted_values_list = []
        plt.figure(figsize=(10, 5), dpi=300)

        # Compute partial dependence for all models
        for model in model_list:
            predicted_values_A = compute_partial_dependence(model, X_train, feature_index, grid_values)
            predicted_values_list.append(predicted_values_A)

        # Convert to NumPy array for easier slicing
        predicted_values_list = np.array(predicted_values_list)

        # Compute statistics
        quant_min = np.quantile(predicted_values_list, 0.0, axis=0)
        quant_max = np.quantile(predicted_values_list, 1.0, axis=0)
        quant_10 = np.quantile(predicted_values_list, 0.1, axis=0)
        quant_90 = np.quantile(predicted_values_list, 0.9, axis=0)
        mean_vals = np.mean(predicted_values_list, axis=0)
        median_vals = np.median(predicted_values_list, axis=0)

        # Plot shaded quantile ranges
        plt.fill_between(grid_values, quant_min, quant_max, color="lightgray", alpha=0.2, label="Full Range (0–1)")
        plt.fill_between(grid_values, quant_10, quant_90, color="gray", alpha=0.2, label="Quantile Range (0.1–0.9)")

        # Optional: Plot quantile lines
        #plt.plot(grid_values, quant_10, color="black", linestyle="--", linewidth=1, label="0.1 Quantile")
        #plt.plot(grid_values, quant_90, color="black", linestyle="--", linewidth=1, label="0.9 Quantile")

        # Plot mean and median
        plt.plot(grid_values, mean_vals, color="red", linewidth=3, label="Mean")
        plt.plot(grid_values, median_vals, color="blue", linewidth=3, label="Median")

        # Formatting
        plt.xlabel(X_train.columns[feature_index] + ' Values')
        plt.ylabel('Predicted Values')
        title = 'Partial Dependence Plot for ' + X_train.columns[feature_index]
        plt.title(title)
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_folder + title + "_" + name_modifier + ".pdf")
        plt.savefig(plot_folder + title + "_" + name_modifier + ".png")
        plt.show()

        # Mean & median only version
        plt.figure(figsize=(10, 5), dpi=300)
        plt.plot(grid_values, mean_vals, color="red", linewidth=3, label="Mean")
        plt.plot(grid_values, median_vals, color="blue", linewidth=3, label="Median")
        plt.xlabel(X_train.columns[feature_index] + ' Values')
        plt.ylabel('Predicted Values')
        plt.title(title)
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_folder + title + "_" + name_modifier + "_mm.pdf")
        plt.savefig(plot_folder + title + "_" + name_modifier + "_mm.png")
        plt.show()

    return print("Plots are saved in " + plot_folder)

# In[ ]:





# In[ ]:


pd.set_option("display.precision", 8)


# # Author
# 
# * Author: Martin Wegmann
# 
# * Date: February 2024
# 
# * Contact: martinwegmann@pm.me


# # Goal

# There are two dependent, dichotomous variables, called: mdr and susceptible.
# 
# (Or if you want to play even more, you could try nres but for this one we would need a Poisson regression…)

# # Folder

# In[ ]:


input_folder="/storage/homefs/mawegmann/vero/input_data/"
plot_folder="/storage/homefs/mawegmann/vero/plots/"


# # Data



data=pd.read_csv(input_folder+"MICcalves_weather_v6_modified.csv",sep=",")


# In[ ]:


input_data=data






# In[ ]:


items_2_remove_A=['probenid2', 'mdr','nwt']
items_2_remove_B=['FarmID','probenid2', 'mdr','nwt']

#items_2_remove_A=['probenid2', 'mdr','nwt', 'Stall', 'nbfarms22', 'pneum1', 'veal', 'diarrh301']
#items_2_remove_B=['FarmID','probenid2', 'mdr','nwt', 'Stall', 'nbfarms22', 'pneum1', 'veal', 'diarrh301']


df_11_list=[]
df_22_list=[]

acc_A_list=[]
acc_B_list=[]

prec_A_list=[]
prec_B_list=[]

rec_A_list=[]
rec_B_list=[]

model_A_list=[]
model_B_list=[]

data_A_list=[]
data_B_list=[]

for i in range(10000):
    print(i)
    with HiddenPrints():
        df_A,df_B,acc_A,acc_B,prec_A,prec_B,rec_A,rec_B,model_A,model_B,data_A,data_B=run_two_rf_BS(data,group_by='probenid2',target="mdr",vars_2_rm_A=items_2_remove_A,vars_2_rm_B=items_2_remove_B,trainvalsplit=.2)
    df_11_list.append(df_A)
    df_22_list.append(df_B)
    acc_A_list.append(acc_A)
    acc_B_list.append(acc_B)
    prec_A_list.append(prec_A)
    prec_B_list.append(prec_B)
    rec_A_list.append(rec_A)
    rec_B_list.append(rec_B)
    model_A_list.append(model_A)
    #model_B_list.append(model_B)


# In[ ]:


#choice=[0,1,2,15,17,18,19,20,21,22,23,24]
choice=[1,2,17,18,19,20,21,22]

# In[ ]:





# In[ ]:


plot_feature_direction_grayspace(data=data_A,var_choice=choice,model_list=model_A_list,plot_folder=plot_folder,name_modifier="MICcalves_weather_mdr",grid_points=50)


# In[ ]:

df_11_list=[]
df_22_list=[]

acc_A_list=[]
acc_B_list=[]

prec_A_list=[]
prec_B_list=[]

rec_A_list=[]
rec_B_list=[]

model_A_list=[]
model_B_list=[]

data_A_list=[]
data_B_list=[]

for i in range(10000):
    print(i)
    with HiddenPrints():
        df_A,df_B,acc_A,acc_B,prec_A,prec_B,rec_A,rec_B,model_A,model_B,data_A,data_B=run_two_rf_BS(data,group_by='probenid2',target="nwt",vars_2_rm_A=items_2_remove_A,vars_2_rm_B=items_2_remove_B,trainvalsplit=.2)
    df_11_list.append(df_A)
    df_22_list.append(df_B)
    acc_A_list.append(acc_A)
    acc_B_list.append(acc_B)
    prec_A_list.append(prec_A)
    prec_B_list.append(prec_B)
    rec_A_list.append(rec_A)
    rec_B_list.append(rec_B)
    model_A_list.append(model_A)
    #model_B_list.append(model_B)


# In[ ]:


#choice=[0,1,2,15,17,18,19,20,21,22,23,24]
choice=[1,2,17,18,19,20,21,22]

# In[ ]:





# In[ ]:


plot_feature_direction_grayspace(data=data_A,var_choice=choice,model_list=model_A_list,plot_folder=plot_folder,name_modifier="MICcalves_weather_nwt",grid_points=50)


# In[ ]:


print("welldone")

