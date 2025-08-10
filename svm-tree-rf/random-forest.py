# -*- coding: utf-8 -*-
"""
@author: jhd9252

Spam email classification using Random Forests 
Using accuracies reported across a number of seeds in:
(1) Gini Impurity 
(2) Shannon Information Gain
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

"""
Data Extraction and Setup:
"""
f = open("spambase.names")
cols = []
for line in f:
    cols.append(line)
f.close()

cols = cols[33:]
cols.append('1 spam 2 non-spam')

for col in range(0, len(cols)):
    split = cols[col].split(":")[0]
    cols[col] = split

df = pd.read_csv("spambase.data", names = cols)
# 4601 observations, 58 columns
assert df.shape == (4601, 58) 
df_desc = df.describe()
df_corr = df.corr().round(2)
sns.heatmap(data=df_corr, annot=False, xticklabels = False, yticklabels = False)

"""
Data Cleaning
"""
# check the data for missing values --> NO MISSING DATA
missing_data = df.isna().sum().sum()
total_entries = df.shape[0] * df.shape[1]
percentage_missing_entry = missing_data / total_entries
print('Total entries (observations * features: ', total_entries)
print('Total missing entries: ', missing_data)
print('Percentage of dataset that is missing: ', percentage_missing_entry)

"""
Data Pre-Processing
"""

seeds = [x for x in range(20)]
estimators = [1,3,5,10,15,20,40,70]
gini = [] 
shannon = []

for estimator in estimators:
    gini_res = []
    shannon_res = []
    for seed in seeds:
        # Split into training and test sets based on seed
        X_train, X_test, Y_train, Y_test = train_test_split(df.iloc[:,:57], 
                                                            df.iloc[:,57], 
                                                            test_size = 0.3, 
                                                            random_state=(seed))
        
        # reshape the X sets
        X_train = X_train.values.reshape((X_train.shape[0],57))
        X_test = X_test.values.reshape((X_test.shape[0], 57))
        
        # reshape the Y sets
        # Y_train = Y_train.values.reshape((Y_train.shape[0],1))
        # Y_test = Y_test.values.reshape((Y_test.shape[0], 1))
        
        model1 = RandomForestClassifier(n_estimators=estimator,
                                        criterion='gini', 
                                        max_depth=None, 
                                        min_samples_split=2, 
                                        min_samples_leaf=1, 
                                        min_weight_fraction_leaf=0.0,
                                        max_features='sqrt', 
                                        max_leaf_nodes=None, 
                                        min_impurity_decrease=0.0, 
                                        bootstrap=True, 
                                        oob_score=False, 
                                        n_jobs=None, 
                                        random_state=seed, 
                                        verbose=0, 
                                        warm_start=False, 
                                        class_weight=None, 
                                        ccp_alpha=0.0, 
                                        max_samples=None)
        
        model2 = RandomForestClassifier(n_estimators=estimator,
                                        criterion='entropy', 
                                        max_depth=None, 
                                        min_samples_split=2, 
                                        min_samples_leaf=1, 
                                        min_weight_fraction_leaf=0.0,
                                        max_features='sqrt', 
                                        max_leaf_nodes=None, 
                                        min_impurity_decrease=0.0, 
                                        bootstrap=True, 
                                        oob_score=False, 
                                        n_jobs=None, 
                                        random_state=seed, 
                                        verbose=0, 
                                        warm_start=False, 
                                        class_weight=None, 
                                        ccp_alpha=0.0, 
                                        max_samples=None)
        model1.fit(X_train, Y_train)
        model2.fit(X_train, Y_train)
        gini_res.append(model1.score(X_test, Y_test))
        shannon_res.append(model2.score(X_test, Y_test))
    
    gini.append(max(gini_res).round(5))
    shannon.append(max(shannon_res).round(5))
        
print("Best Test Accuracy with Gini Impurity Criterion for each #Estimator, each across 20 different seeds: ")
print(gini)
print("Best Test Accuracy with Shannon I.G Criterion for each #Estimator, each across 20 different seeds: ")
print(shannon)