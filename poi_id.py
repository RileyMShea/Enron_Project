#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function

import pickle
import sys

sys.path.append("../tools/")
import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from tester import dump_classifier_and_data, test_classifier

# Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# replace 'Nan' strings with None in dataset
for outer_keys, inner_dicts in data_dict.items():
    for k, v in inner_dicts.items():
        if v == 'NaN':
            data_dict[outer_keys][k] = None

df = pd.DataFrame.from_dict(data_dict,
                            orient='index'  # user outer dict keys as column names
                            )
# Handles email_address field
df.fillna(value=pd.np.nan, inplace=True)

# the 'TOTAL' record
TOTAL = df.sort_values(by=['salary'], ascending=False, na_position='last').head(1)
# dropping computed 'TOTAL' observation
df.drop(index='TOTAL', inplace=True)

# Drop records where all inputs are NaN
df.dropna(thresh=2, inplace=True)

# Not a person, dropping
df.drop(index='THE TRAVEL AGENCY IN THE PARK', inplace=True)

# New Feature - has_email - creation
df['has_email'] = df.email_address.notna()

# create copy of dataframe
X = df.copy()
# Dropping email_adress field
X.drop(['email_address'], axis=1, inplace=True)
# Popping poi field
y = X.pop('poi')

# Impute missing values
X_imputed = Imputer(
    strategy="most_frequent",
    axis=0).fit_transform(X)
imp_df = pd.DataFrame(X_imputed, columns=X.columns, index=X.index)
imp_df['poi'] = y

# Scale values
X_imp_std = StandardScaler().fit_transform(X_imputed)

# Bring values back into a Dataframe
X_imp_std = StandardScaler().fit_transform(X_imputed)
imp_std_df = pd.DataFrame(X_imp_std, columns=X.columns, index=X.index)
imp_std_df['poi'] = y
# convert dataset back to dictionary for tester.py
dict_for_tester = imp_std_df.to_dict('index')
imp_std_df.head()

# Support vector machine testing
# -----------------------------
# Warning supression
np.seterr(divide='ignore', invalid='ignore')
import warnings

warnings.filterwarnings('ignore')

# Building pipeline
pipeline = Pipeline([
    ("features", SelectKBest(f_classif)),
    ("svm", SVC())])

# Paramater grid for gridsearchcv
param_grid = dict(features__k=np.arange(1, X.shape[1]),
                  svm__C=[0.1, 1., 10.],
                  svm__kernel=['linear', 'rbf', 'sigmoid']
                  )

grid_search = GridSearchCV(estimator=pipeline,
                           param_grid=param_grid,
                           scoring='f1',
                           cv=5,
                           iid=True)

grid_search.fit(X_imp_std, y)

# the classifier used
clf = grid_search.best_estimator_.steps[1][1]
# features selected
cols = grid_search.best_estimator_.steps[0][1].get_support(indices=True)
feature_list = ['poi'] + list(X.columns[cols])

# Run test classifier from tester.py
test_classifier(clf,
                dataset=dict_for_tester,
                feature_list=feature_list)

# Gaussian Naive Bayes testing
# -----------------------------

# Building pipeline
pipeline = Pipeline([
    ("features", SelectKBest(f_classif)),
    ("gnb", GaussianNB())])

# Paramater grid for gridsearchcv
param_grid = dict(features__k=np.arange(1, X.shape[1]))

grid_search = GridSearchCV(estimator=pipeline,
                           param_grid=param_grid,
                           cv=5,
                           iid=True,
                           scoring='f1'
                           )

grid_search.fit(X_imp_std, y)

# the classifier used
clf = grid_search.best_estimator_.steps[1][1]
# features selected
cols = grid_search.best_estimator_.steps[0][1].get_support(indices=True)
feature_list = ['poi'] + list(X.columns[cols])

# Run test classifier from tester.py
test_classifier(clf,
                dataset=dict_for_tester,
                feature_list=feature_list)

# Dump required objects
dump_classifier_and_data(clf,
                         dataset=dict_for_tester,
                         feature_list=feature_list)
