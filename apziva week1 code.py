#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 14:32:49 2024

@author: adedoyinolayanju
"""

import numpy as np
import pandas as pd
data=pd.read_csv('/Users/adedoyinolayanju/Downloads/datap.csv')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
print(data.head())
print(data.describe())
print(data.describe(include=['O']))
print(data.info())
print(data.isnull().sum())
data.hist(bins=20, figsize=(20, 15))
plt.show()
plt.figure(figsize=(20, 15))
for i, column in enumerate(data.select_dtypes(include=['object']).columns, 1):
    plt.subplot(3, 3, i)
    sns.countplot(data=data, x=column)
    plt.title(f'Count plot of {column}')
    plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
sns.pairplot(data.select_dtypes(include=['number']))
plt.show()
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()
plt.figure(figsize=(15, 10))
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
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
