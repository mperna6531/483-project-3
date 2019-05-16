#!/usr/bin/env python3

import pandas as pd
import numpy as np

import sklearn.preprocessing

# Experiment 0: preprocess data - code provided from Prof Kenytt Avery
# bank.csv from https://archive.ics.uci.edu/ml/datasets/Bank+Marketing


def load_data(csv):
    # Ordinal features
    #
    # Note: month probably shouldn't be ordinal.
    # Then again, neither should day.
    bank = pd.read_csv(csv, sep=';')
    boolean = {'no': 0.0, 'yes': 1.0}
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
    bank = pd.get_dummies(bank, columns=categorical,
                        prefix=categorical, drop_first=True)

    # Numeric features
    #
    # Standardized because we plan to use KNN and SVM

    scaled = ['age', 'balance', 'day', 'month',
            'duration', 'campaign', 'pdays', 'previous']
    bank[scaled] = sklearn.preprocessing.scale(bank[scaled].astype(float))

    # Training set and targets
    X = bank.drop(columns='y').values
    t = bank['y'].values

    return X, t
