# -*- coding: utf-8 -*-
"""
@author: jhd9252

Trying support vector machines to classify spam or non-spam categories. 
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

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
# check the data for missing values --> NO MISSING DATA
missing_data = df.isna().sum().sum()
total_entries = df.shape[0] * df.shape[1]
percentage_missing_entry = missing_data / total_entries
print('Total entries (observations * features: ', total_entries)
print('Total missing entries: ', missing_data)
print('Percentage of dataset that is missing: ', percentage_missing_entry)

"""
Data Pre-Processing
1. Combine or remove any unnecessary rows - unnecessary, SVM work well with higher dimensions
2. Split the data 70:30 - done
3. Choose normalization or standardization with justification -> normalization
4. dimensionality reduction -> not needed, SVM works well in higher dimensions with hyper plane. Perhaps non-linear data. No assumptions. 
"""

# Split into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(df.iloc[:,:57], df.iloc[:,57], test_size = 0.3, random_state=(0))

# reshape the X sets
X_train = X_train.values.reshape((X_train.shape[0],57))
X_test = X_test.values.reshape((X_test.shape[0], 57))

# reshape the Y sets
Y_train = Y_train.values.reshape((Y_train.shape[0],1))
Y_test = Y_test.values.reshape((Y_test.shape[0], 1))

# confirm the shapes of sets
print("X_train: ", X_train.shape)
print("X_test:  ", X_test.shape)
print("Y_train: ", Y_train.shape)
print("Y_test:  ", Y_test.shape)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

"""
Modeling: SVMs
"""

# list of C parameters
c_vals = [10**(-2), 10**(-1), 10**(0), 10**(1), 10**(2), 10**(3), 10**(4)]

# Linear accruacy, tuples(train accuracy, test accuracy) for each C
linear_accur = []

# quadratic acc, 2 lists
quad_accur = []

# rbf accur, 2 lists
rbf_accur = []

# predict
for c in c_vals:
    model_linear = SVC(C=c, kernel = "linear")
    model_linear.fit(X_train, Y_train) 
    score_linear_train = model_linear.score(X_train, Y_train).round(4)
    score_linear_test = model_linear.score(X_test, Y_test).round(4)
       
    model_quad = SVC(C=c, kernel = "poly", degree = 2)
    model_quad.fit(X_train, Y_train)
   
    score_quad_train = model_quad.score(X_train, Y_train).round(4)
    score_quad_test = model_quad.score(X_test, Y_test).round(4)
       
    model_rbf = SVC(C=c, kernel = "rbf", gamma = "scale")
    model_rbf.fit(X_train, Y_train)
    score_rbf_train = model_rbf.score(X_train, Y_train).round(4)
    score_rbf_test = model_rbf.score(X_test, Y_test).round(4)
       
    linear_accur.append((score_linear_train, score_linear_test))
    quad_accur.append((score_quad_train, score_quad_test))
    rbf_accur.append((score_rbf_train, score_rbf_test))
    
lin_train = [x[0] for x in linear_accur]
lin_test = [x[1] for x in linear_accur]
quad_train =[x[0] for x in quad_accur]
quad_test =[x[1] for x in quad_accur]
rbf_train =[x[0] for x in rbf_accur]
rbf_test =[x[1] for x in rbf_accur]

res_minmaxscalar = [lin_train, lin_test, quad_train, quad_test, rbf_train, rbf_test]
df_res = pd.DataFrame(res_minmaxscalar)
    
c_vals = [str(c) for c in c_vals]

sns.heatmap(df_res, annot = True, xticklabels = c_vals, yticklabels=['Linear','','Quadratic','','RBF',''])







