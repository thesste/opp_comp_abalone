#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 21:47:17 2019

@author: Steffen Thesdorf
"""

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

from mord import LogisticAT, LogisticIT

from opp import opp_onevsnext_predict, opp_onevsfollowers_predict, orderedpartitions_predict
from perf import performance_summary

#%%

### Data Preparation ###

source = 'http://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data'

column_names = ['sex',
                'length',
                'diameter',
                'height',
                'whole_weight',
                'shucked_weight',
                'viscera_weight',
                'shell_weight',
                'rings']

df = pd.read_csv(source, names=column_names)  

# Convert from Categorial to Numeric Input  

df = df.assign(male = (df.sex=='M').astype('int32'),
               infant = (df.sex=='I').astype('int32'))

# Separate Labels from Input

X, y = df.drop(columns=['sex', 'rings']), df.rings

#%%

Models = []
Summary = []
no_iter = 10

Classifier = [LinearDiscriminantAnalysis(),\
              SVC(kernel='linear'),\
              MLPClassifier(hidden_layer_sizes=(100, 100)),\
              RandomForestClassifier(min_samples_leaf=5),\
              GradientBoostingClassifier()]

os.makedirs('Output', exist_ok=True)

for i in range(1, no_iter+1):

    # Split Data
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Bin Tail Output Values and Shift
    
    lower_bound = 3
    upper_bound = 24
    
    y_train_replaced_1 = y_train.where(y_train >= lower_bound, lower_bound)
    y_train_replaced_2 = y_train_replaced_1.where(y_train <= upper_bound, upper_bound)
    
    y_train_shifted = y_train_replaced_2 - (lower_bound - 1)
    
    #%%    
    
    ### Model Estimation ###
       
    for clf in [LogisticAT(), LogisticIT()]:
        
        model_str = str(clf)[:10]    
    
        if model_str not in Models:
            Models += [model_str]
            
        y_predict_shifted = clf.fit(X_train, y_train_shifted).predict(X_test)
        y_predict = y_predict_shifted + (lower_bound - 1)
        
        Results, Measures = performance_summary(y_predict, y_test, conf=True, conf_label='Output/'+model_str)    
        Summary += Results
        print(model_str, list(zip(Measures, Results)))
    
    #%%

    for clf in Classifier:
        
        model_str = str(clf)[:10]    
    
        if model_str not in Models:
            Models += [model_str]
            
        y_predict = clf.fit(X_train, y_train).predict(X_test)
        
        Results, Measures = performance_summary(y_predict, y_test, conf=True, conf_label='Output/'+model_str)    
        Summary += Results
        print(model_str, list(zip(Measures, Results)))
            
    #%%
    
    for prefix, opp_func in list(zip(['OVN', 'OVF', 'OP'],
                                 [opp_onevsnext_predict,
                                  opp_onevsfollowers_predict,
                                  orderedpartitions_predict,])):
    
        for forward_flag in [True, False]:
        
            for clf in Classifier:
        
                model_str = str(clf)[:3] + '_' + prefix + '_' + forward_flag*'FWD' + (1-forward_flag)*'BWD'  
            
                if model_str not in Models:
                    Models += [model_str]
                    
                y_predict_shifted = opp_func(X_train, y_train_shifted, clf, X_test, forward=forward_flag)
                y_predict = y_predict_shifted + (lower_bound - 1)
    
                Results, Measures = performance_summary(y_predict, y_test, conf=True, conf_label='Output/'+model_str)   
                Summary += Results
                print(model_str, list(zip(Measures, Results)))
            
#%%
            
### Performance Evaluation ###

Summary = np.array(Summary).reshape([no_iter, len(Models), len(Measures)])

Summary_mean = pd.DataFrame(Summary.mean(axis=0), index=Models, columns=Measures)
Summary_std = pd.DataFrame(Summary.std(axis=0), index=Models, columns=Measures)

Summary_mean.to_csv('Output/modeleval_means.csv')
Summary_std.to_csv('Output/modeleval_std.csv')