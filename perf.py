#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 22:03:26 2019

@author: roger
"""

import numpy as np
from sklearn import metrics

################################################################################################  
    
def AMAE(y_predict, y_test_select):   
    Actual_classes = list(set(y_test_select))
    abs_dev = np.abs(y_test_select - y_predict)
    AMAE_temp = 0    
    
    for c in Actual_classes:
        c_obs = (y_test_select == c).sum()
        AMAE_temp += (1/c_obs)*(abs_dev*(y_test_select == c)).sum()
        
    return (1/len(Actual_classes))*AMAE_temp

################################################################################################

def MAE(y_predict, y_test_select):    
    return np.abs(y_test_select - y_predict).mean()
    
################################################################################################  

def MSE(y_predict, y_test_select):    
    return ((y_test_select - y_predict)**2).mean()
    
################################################################################################

def performance_summary(y_predict, y_test_select, conf=False, conf_label=''):

    score = (y_predict == y_test_select).sum() / len(y_test_select)
    one_notch_score = (np.abs(y_predict - y_test_select) <= 1).sum() / len(y_test_select)
    two_notch_score = (np.abs(y_predict - y_test_select) <= 2).sum() / len(y_test_select)
    amae = AMAE(y_predict, y_test_select)
    mae = MAE(y_predict, y_test_select)
    mse = MSE(y_predict, y_test_select)
    
    results = [score, one_notch_score, two_notch_score, amae, mae, mse]
    measures = ['Score', '1-Notch Score', '2-Notches Score', 'AMAE', 'MAE', 'MSE']     
    
    if conf:
        np.savetxt(conf_label + '_conf_matrix.csv', metrics.confusion_matrix(y_test_select, y_predict), delimiter=',')
        
    return results, measures