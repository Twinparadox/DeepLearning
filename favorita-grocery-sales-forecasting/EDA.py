# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 14:10:55 2020

@author: nww73
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_train = pd.read_csv('./data/train.csv', engine='c')
df_test = pd.read_csv('./data/test.csv', engine='c')

# pd.write_csv(df_train, './data/train_drop.csv')

train_store_nbr_freq = df_train['store_nbr'].value_counts()
train_item_nbr_freq = df_train['item_nbr'].value_counts()
train_unit_sales = df_train['unit_sales'].value_counts()

test_store_nbr_freq = df_test['store_nbr'].value_counts()
test_item_nbr_freq = df_test['item_nbr'].value_counts()

bins = train_store_nbr_freq.shape[0]
df_train['store_nbr'].plot.hist(bins=bins)
plt.show()
bins = test_store_nbr_freq.shape[0]
df_test['store_nbr'].plot.hist(bins=bins)
plt.show()
bins = train_item_nbr_freq.shape[0]
df_train['item_nbr'].plot.hist(bins=bins)
plt.ylim()
plt.show()
bins = test_item_nbr_freq.shape[0]
df_test['item_nbr'].plot.hist(bins=bins)
plt.ylim(0,)
plt.show()
bins = int(train_unit_sales.shape[0]/100)
df_train['unit_sales'].plot.hist(bins=bins)
plt.ylim()
plt.show()