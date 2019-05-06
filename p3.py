#!/usr/bin/env python3 

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

import sklearn.preprocessing

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

# Experiment 0: preprocess data - code provided from Prof Kenytt Avery
# bank.csv from https://archive.ics.uci.edu/ml/datasets/Bank+Marketing

bank = pd.read_csv('bank.csv', sep=';')

# Ordinal features
#
# Note: month probably shouldn't be ordinal.
# Then again, neither should day.

boolean = { 'no': 0.0, 'yes': 1.0 }
months = {
    'jan': 1.0, 'feb': 2.0, 'mar': 3.0, 'apr': 4.0,  'may': 5.0,  'jun': 6.0,
    'jul': 7.0, 'aug': 8.0, 'sep': 9.0, 'oct': 10.0, 'nov': 11.0, 'dec': 12.0
}

bank.replace({
    'default': boolean,
    'housing': boolean,
    'loan':    boolean,

    'month':   months,
    'y':       boolean
}, inplace=True)

# Categorical features
#
# Since we plan to use logistic regression, add drop_first=True
# to use dummy instead of one-hot encoding

categorical = ['job', 'marital', 'education', 'contact', 'poutcome']
bank = pd.get_dummies(bank, columns=categorical, prefix=categorical, drop_first=True)

# Numeric features
#
# Standardized because we plan to use KNN and SVM 

scaled = ['age', 'balance', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous']
bank[scaled] = sklearn.preprocessing.scale(bank[scaled].astype(float))

# Training set and targets
X = bank.drop(columns='y').values
t = bank['y'].values


# Experiment 1
X_train, X_test, y_train, y_test = train_test_split(X, t,
                            test_size=0.2, shuffle=True, random_state=1) 

# Experiment 2
clf = GaussianNB()
clf.fit(X_train, y_train)

# Experiment 3
#
# a
print('Score: ', clf.score(X_test, y_test))

# b
y_pred = clf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

# c, d 
# 

# needs double checking

y_probs = clf.predict_proba(X_test)
fpr, tpr, thresholds = roc_curve(y_test, y_probs[:,1])
auc = roc_auc_score(y_test, y_probs[:,1])

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label=f'ROC curve, AUC = {auc:.3f}')
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic for GaussianNB')
plt.legend(loc="lower right")
plt.show()


# Experiment 4
linreg_clf = LogisticRegression(fit_intercept=False, solver='lbfgs', random_state=1)
linreg_clf.fit(X_train, y_train)

# Experiment 5
kernels = ['linear', 'poly', 'rbf', 'sigmoid']

results = dict()
for kernel in kernels:
    svc = SVC(kernel=kernel, gamma='scale', random_state=1).fit(X_train, y_train)
    results[kernel] = svc.score(X_test, y_test)

print('Best performing kernel: ', max(zip(results.values(), results.keys())))
