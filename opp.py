# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 09:56:23 2018

@author: Steffen Thesdorf
"""

import numpy as np

def opp_onevsfollowers_predict(X_train, y_train, bin_class, X_test, forward=True):
        
    assert len(set(y_train)) == max(y_train)

    y_predict = np.zeros([len(X_test)], dtype=int)    
    
    X_train_temp = X_train
    y_train_temp = y_train
        
    if forward:
        
        for grade in range(1, max(y_train)):
            
            y_predict_grade = bin_class.fit(X_train_temp, (y_train_temp == grade).astype('int')).predict(X_test)
            y_predict += y_predict_grade*(y_predict == 0)*grade
                           
            X_train_temp = X_train_temp[y_train_temp != grade]
            y_train_temp = y_train_temp[y_train_temp != grade]
            
        y_predict += max(y_train)*(y_predict == 0)
    
    else:    
    
        for grade in list(reversed(range(2, max(y_train) + 1))):
            
            y_predict_grade = bin_class.fit(X_train_temp, (y_train_temp == grade)).predict(X_test)
            y_predict += y_predict_grade*(y_predict == 0)*grade
                           
            X_train_temp = X_train_temp[y_train_temp != grade]
            y_train_temp = y_train_temp[y_train_temp != grade]
            
        y_predict += (y_predict == 0)
    
    return y_predict
    
########################################################################################################    
    
def opp_onevsnext_predict(X_train, y_train, bin_class, X_test, forward=True):

    assert len(set(y_train)) == max(y_train)

    y_predict = np.zeros([len(X_test)], dtype=int)    
    
    if forward:
        
        for grade in range(1, max(y_train)):
            
            X_train_temp = X_train[((y_train == grade) | (y_train == grade + 1))]
            y_train_temp = y_train[((y_train == grade) | (y_train == grade + 1))]
                    
            y_predict_grade = bin_class.fit(X_train_temp, (y_train_temp == grade)).predict(X_test)
            y_predict += y_predict_grade*(y_predict == 0)*grade
            
        y_predict += max(y_train)*(y_predict == 0)
        
    else:
        
        for grade in list(reversed(range(2, max(y_train) + 1))):
            
            X_train_temp = X_train[((y_train == grade) | (y_train == grade - 1))]
            y_train_temp = y_train[((y_train == grade) | (y_train == grade - 1))]
                
            y_predict_grade = bin_class.fit(X_train_temp, (y_train_temp == grade)).predict(X_test)
            y_predict += y_predict_grade*(y_predict == 0)*grade
            
        y_predict += (y_predict == 0)
    
    return y_predict
    
########################################################################################################
    
def orderedpartitions_predict(X_train_select, y_train_select, bin_class, X_test_select, forward=True):     
    
    assert len(set(y_train_select)) == max(y_train_select)
    
    y_predict = np.zeros([len(X_test_select)], dtype=int)
    
    if forward:    
    
        for grade in range(1, max(y_train_select)):
                    
            y_predict_grade = bin_class.fit(X_train_select, (y_train_select <= grade)).predict(X_test_select)
            y_predict += y_predict_grade*(y_predict == 0)*grade
        
        y_predict += max(y_train_select)*(y_predict == 0)
    
    else:
    
        for grade in list(reversed(range(2, max(y_train_select) + 1))):
                    
            y_predict_grade = bin_class.fit(X_train_select, (y_train_select >= grade)).predict(X_test_select)
            y_predict += y_predict_grade*(y_predict == 0)*grade
            
        y_predict += (y_predict == 0)
    
    return y_predict