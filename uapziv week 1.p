#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 11:55:57 2024

@author: adedoyinolayanju
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
data=pd.read_csv('/Users/adedoyinolayanju/Downloads/datap.csv')
data
X = data.drop('Y', axis=1)
y = data['Y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
print(model.coefficients)
import statsmodels.api as sm
model = sm.Logit(y_train, X_train)
result = model.fit()
print(result.summary())