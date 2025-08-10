# -*- coding: utf-8 -*-
"""
@author: jhd9252

Trying decision trees to classify spam or non-spam categories. 
Find the best classification accuracy across two different seeds.
Results are using (1) Gini Impurity (2) Shannon Information Gain which uses the idea of entropy
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

"""
Data Extraction and Setup:
1. Download the data set, open as a file object
2. grab the column names
3. slice the column names ->get rid after colon --> get rid colon
4. grab the dataset and create dataframe
5. get the basic descriptive statistics
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
[x] Clean Data
[x] lowercases
[x] removing special charactesr
[x] remove stopwords
[x] remove hyperlinks
[x] remove numbers if needed
[x] remove whitespaces
[x] check for missing data / decide how to deal with it
"""

missing_data = df.isna().sum().sum()
total_entries = df.shape[0] * df.shape[1]
percentage_missing_entry = missing_data / total_entries
print('Total entries (observations * features: ', total_entries)
print('Total missing entries: ', missing_data)
print('Percentage of dataset that is missing: ', percentage_missing_entry)

"""
Data Pre-Processing
1. Combine or remove any unnecessary rows - unnecessary
2. Split the data 70:30 - done
3. Choose normalization or standardization with justification -> None
4. dimensionality reduction -> Maybe? 
"""

# Note: Decision Trees and RF generally do not deal with distance calculations
# or relations between points in space. Rather they deal with magnitude or order.
# Therefore no regularization or standardization is strictly necessary. 

# preprocessing based on seed, model and score based on seed
seeds = [x for x in range(20)]
gini = [0] * 20
shannon = [0]*20

for seed in seeds:        
    # Split into training and test sets based on seed
    X_train, X_test, Y_train, Y_test = train_test_split(df.iloc[:,:57], df.iloc[:,57], test_size = 0.3, random_state=(seed))
    
    # reshape the X sets
    X_train = X_train.values.reshape((X_train.shape[0],57))
    X_test = X_test.values.reshape((X_test.shape[0], 57))
    
    # reshape the Y sets
    Y_train = Y_train.values.reshape((Y_train.shape[0],1))
    Y_test = Y_test.values.reshape((Y_test.shape[0], 1))
    
    # models
    model1 = DecisionTreeClassifier(
                                   criterion='gini', 
                                   splitter='best', 
                                   max_depth=None, 
                                   min_samples_split=2, 
                                   min_samples_leaf=1, 
                                   min_weight_fraction_leaf=0.0, 
                                   max_features=None, 
                                   random_state=seed, 
                                   max_leaf_nodes=None, 
                                   min_impurity_decrease=0.0, 
                                   class_weight=None, 
                                   ccp_alpha=0.0)
    model2 = DecisionTreeClassifier(
                                   criterion='entropy', 
                                   splitter='best', 
                                   max_depth=None, 
                                   min_samples_split=2, 
                                   min_samples_leaf=1, 
                                   min_weight_fraction_leaf=0.0, 
                                   max_features=None, 
                                   random_state=seed, 
                                   max_leaf_nodes=None, 
                                   min_impurity_decrease=0.0, 
                                   class_weight=None, 
                                   ccp_alpha=0.0)
    
    model1.fit(X_train, Y_train)
    model2.fit(X_train, Y_train)
    
    gini[seed] = (model1.score(X_test, Y_test).round(5))
    shannon[seed] = (model2.score(X_test, Y_test).round(5))

print("Best Test Accuracy for Gini criterion: ", max(gini))
print("Best Test Accuracy for Shannon I.G criterion: ", max(shannon))