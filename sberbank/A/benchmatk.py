#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 00:46:43 2016

@author: ottogin
"""

import pandas as pd
import numpy as np

from sklearn.ensemble import GradientBoostingClassifier

transactions = pd.read_csv('../Data/pr_trans')
customers_gender = pd.read_csv('../Data/customers_gender_train.csv')
humans_tr = pd.read_csv('../Data/homan_trans', index_col=0)

grtr = transactions.groupby('customer_id')

Xmcc = grtr.apply(lambda x: x[['mcc_code']].unstack().value_counts()) \
												.unstack().fillna(0)
Xmcc.columns = ['mcc%d'%d for d in Xmcc.columns]
	
	
Xday = grtr.apply(lambda x: x[['day']].unstack().value_counts()) \
												.unstack().fillna(0)
Xday.columns = ['day%d'%d for d in Xday.columns]


Xtrt = grtr.apply(lambda x: x[['tr_type']].unstack().value_counts()) \
												.unstack().fillna(0)
Xtrt.columns = ['trt%d'%d for d in Xtrt.columns]

X = pd.concat([Xday, Xmcc, Xtrt, humans_tr], axis=1)

print(X)

customers_gender = customers_gender.set_index('customer_id')

X_train = X.loc[customers_gender.index].reset_index()
Y_train = customers_gender.gender.reset_index()
print(Y_train, X_train)


clf = GradientBoostingClassifier(random_state=13)
clf.fit(X_train, Y_train.values[:, 0])

X_test = X.drop(customers_gender.index)

result = pd.DataFrame(X_test.index, columns=['customer_id'])
result['gender'] = clf.predict_proba(X_test)[:, 1]

result.to_csv('sol/morefeatures.csv', index=False)
