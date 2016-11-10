#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 10:59:08 2016

@author: ottogin
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import LinearRegression
from scipy import sparse
from itertools import product

transactions = pd.read_csv('../Data/transactions.csv')
customers_gender = pd.read_csv('../Data/customers_gender_train.csv')

cuses_test = list(set(transactions.customer_id.unique().tolist())\
                  .difference(customers_gender.customer_id.unique()))
all_cuses = transactions.customer_id.unique()
all_mcc = transactions.mcc_code.unique()

transactions = transactions[transactions.amount < 0].copy()
transactions['day'] = transactions.tr_datetime.apply(lambda dt: \
                                                    dt.split()[0]).astype(int)

transactions.day += 29 - transactions['day'].max()%30
transactions['month_num'] = (transactions.day) // 30
transactions['year_num'] = (transactions.day) // 365

test_transactions = transactions[transactions.month_num == 15]
train_transactions = transactions[transactions.month_num < 15]

test_transactions = test_transactions.set_index('customer_id')
test_transactions = test_transactions.loc[cuses_test]
test_transactions = test_transactions.reset_index()

grid = list(product(*[all_cuses, all_mcc, range(10, 15)]))
train_grid = pd.DataFrame(grid, columns = ['customer_id', 'mcc_code',\
                                                                  'month_num'])

test_grid = list(product(*[cuses_test, all_mcc]))       
test_grid = pd.DataFrame(test_grid, columns = ['customer_id', 'mcc_code'])
test_grid['month_num'] = 15

test = pd.merge(test_grid,
         test_transactions.groupby(['year_num', 'month_num',\
                'customer_id', 'mcc_code'])[['amount']].sum().reset_index(),
         how='left').fillna(0)
         
train = pd.merge(train_grid,
         train_transactions.groupby(['year_num', 'month_num', \
                'customer_id', 'mcc_code'])[['amount']].sum().reset_index(),
         how='left').fillna(0)
         
for month_shift in range(1, 3):
    train_shift = train.copy()
    train_shift['month_num'] = train_shift['month_num'] + month_shift
    train_shift = train_shift.rename(columns={"amount" : \
                                             'amount_{0}'.format(month_shift)})  
    train_shift = train_shift[['year_num', 'month_num', 'customer_id',\
                               'mcc_code', 'amount_{0}'.format(month_shift)]]

    train = pd.merge(train, train_shift, 
                                  on=['year_num', 'month_num',\
                            'customer_id', 'mcc_code'], how='left').fillna(0)
    test = pd.merge(test, train_shift, 
                                 on=['year_num', 'month_num', \
                            'customer_id', 'mcc_code'], how='left').fillna(0)
hasher = FeatureHasher(n_features=10000, input_type='string')
train_sparse = \
    hasher.fit_transform(train[['year_num', 'month_num', \
                        'customer_id', 'mcc_code']].astype(str).as_matrix())

test_sparse = \
    hasher.transform(test[['year_num', 'month_num', 'customer_id',\
                                       'mcc_code']].astype(str).as_matrix())
 
train_sparse = sparse.hstack([train_sparse,
            np.log(np.abs(train[['amount_1', 'amount_2']]) + 1).as_matrix()])

test_sparse = sparse.hstack([test_sparse,
            np.log(np.abs(test[['amount_1', 'amount_2']]) + 1).as_matrix()])

shift = 1
clf = LinearRegression()
clf.fit(train_sparse, np.log(-train['amount'] + shift))
test['volume'] = np.e ** clf.predict(test_sparse) - shift
test[['customer_id', 'mcc_code', 'volume']].to_csv('sol/baseline_c.csv'\
                                                                , index=False)    