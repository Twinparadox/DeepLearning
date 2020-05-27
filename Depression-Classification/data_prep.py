# -*- coding: utf-8 -*-
"""
Created on Wed May 27 15:49:50 2020

@author: wonwoo
"""
import pandas as pd
import numpy as np

INQ = ['INQ020', 'INQ012', 'INQ030', 'INQ060', 'INQ080', 'INQ090', 'INQ132',
       'INQ140', 'INQ150', 'IND235']
DPQ = ['DPQ010', 'DPQ020', 'DPQ030', 'DPQ040', 'DPQ050', 'DPQ060', 'DPQ070',
       'DPQ080', 'DPQ090', 'DPQ_range', 'DPQ_OHE']
DEMO = ['RIAGENDR']
column_list = INQ+DPQ+DEMO


df_2007 = pd.read_csv('2007-2008.csv', engine='c')
df_2009 = pd.read_csv('2009-2010.csv', engine='c')
df_2011 = pd.read_csv('2011-2012.csv', engine='c')
df_2013 = pd.read_csv('2013-2014.csv', engine='c')

df_data = pd.concat([df_2007, df_2009, df_2011, df_2013])

print(df_data.shape)
df_data[column_list] = df_data[column_list].astype(int)
df_data = pd.get_dummies(df_data, columns=column_list)
print(df_data.shape)

df_data = df_data.set_index('SEQN')