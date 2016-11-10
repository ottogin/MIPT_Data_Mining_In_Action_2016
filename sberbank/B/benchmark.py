#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 10:30:54 2016

@author: ottogin
"""

import pandas as pd
import numpy as np

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression

transactions = pd.read_csv('../Data/transactions.csv')
train_transactions = transactions[transactions.amount < 0].copy()
train_transactions['day'] = train_transactions.tr_datetime\
                                .apply(lambda dt: dt.split()[0]).astype(int)
                                
test_transactions = pd.DataFrame(columns=train_transactions.mcc_code.unique(), 
                                 index=np.arange(1, 31) + train_transactions.day.max())
test_transactions = test_transactions.unstack().reset_index().dropna(axis=1)
test_transactions.columns = ['mcc_code', 'day']
train_grid = pd.DataFrame(columns=train_transactions.mcc_code.unique(), 
                          index=train_transactions.day.unique())
train_grid = train_grid.unstack().reset_index().dropna(axis=1)
train_grid.columns = ['mcc_code', 'day']

for tr_table in [train_transactions, test_transactions, train_grid]:
    tr_table['week_num'] = tr_table['day'] // 7
    tr_table['week_day'] = tr_table['day'] % 7
    tr_table['month_num'] = tr_table['day'] // 30
    tr_table['month_day'] = tr_table['day'] % 30

train_transactions = \
    pd.merge(train_grid,
             train_transactions.groupby(['day', 'week_num', 'week_day', \
             'month_num', 'month_day', 'mcc_code'])[['amount']]\
                 .sum().reset_index(),
             how='left').fillna(0)

for day_shift in [-1, 0, 1]:
    for month_shift in train_transactions.month_num.unique()[1:]:
        train_shift = train_transactions.copy()
        train_shift['month_num'] += month_shift
        train_shift['month_day'] += day_shift
        train_shift['amount_day_{}_{}'.format(day_shift, month_shift)] = \
                                        np.log(-train_shift['amount'] + 1)
        train_shift = train_shift[['month_num', 'month_day', 'mcc_code',\
                            'amount_day_{}_{}'.format(day_shift, month_shift)]]

        train_transactions = pd.merge(train_transactions, train_shift, 
                                      on=['month_num', 'month_day', \
                                      'mcc_code'], how='left').fillna(0)
        test_transactions = pd.merge(test_transactions, train_shift, 
                                     on=['month_num', 'month_day', \
                                     'mcc_code'], how='left').fillna(0)

shift = 500
train = pd.get_dummies(train_transactions, columns=['mcc_code'])
test = pd.get_dummies(test_transactions, columns=['mcc_code'])
c = train.columns.difference(['amount'])

clf = LinearRegression()
clf.fit(train[c], np.log(-train['amount'] + shift))

test_transactions['volume'] = np.e ** clf.predict(test[c]) - shift
test_transactions[['mcc_code', 'day', 'volume']].\
                to_csv('sol/baseline_b.csv', index=False)