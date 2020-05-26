# -*- coding: utf-8 -*-
"""
Created on Tue May 26 10:55:21 2020

@author: wonwoo
"""
import numpy as np
import pandas as pd

dir_name = '2013-2014'
init = 'H'

df_CBC = pd.read_sas(dir_name+'/CBC_'+init+'.XPT', index='SEQN')
df_BMX = pd.read_sas(dir_name+'/BMX_'+init+'.XPT', index='SEQN')
df_BPX = pd.read_sas(dir_name+'/BPX_'+init+'.XPT', index='SEQN')
df_INQ = pd.read_sas(dir_name+'/INQ_'+init+'.XPT', index='SEQN')
df_DEMO = pd.read_sas(dir_name+'/DEMO_'+init+'.XPT', index='SEQN')
df_DPQ = pd.read_sas(dir_name+'/DPQ_'+init+'.XPT', index='SEQN')
df_HDL = pd.read_sas(dir_name+'/HDL_'+init+'.XPT', index='SEQN')
df_TCHOL = pd.read_sas(dir_name+'/TCHOL_'+init+'.XPT', index='SEQN')
df_TRIGLY = pd.read_sas(dir_name+'/TRIGLY_'+init+'.XPT', index='SEQN')
df_GHB = pd.read_sas(dir_name+'/GHB_'+init+'.XPT', index='SEQN')

df_total = pd.concat([df_DEMO, df_BMX, df_BPX, df_CBC, df_HDL,
                      df_TCHOL, df_TRIGLY, df_GHB, df_INQ, df_DPQ], axis=1, join='inner')


INQ = ['INQ020', 'INQ012', 'INQ030', 'INQ060', 'INQ080', 'INQ090', 'INQ132',
       'INQ140', 'INQ150', 'IND235']
DPQ = ['DPQ010', 'DPQ020', 'DPQ030', 'DPQ040', 'DPQ050', 'DPQ060', 'DPQ070',
       'DPQ080', 'DPQ090']
DEMO = ['RIAGENDR']

NUMERIC = ['LBXTC', 'LBDTCSI', 'LBDHDD', 'LBDHDDSI', 'LBXGH', 'LBDLDL', 'LBDLDLSI',
           'LBXWBCSI', 'LBXHGB', 'LBXMCHSI', 'LBXMC', 'BMXWT', 'BMXHT', 'BMXBMI',
           'RIDAGEYR', 'BPXPLS']
BP_SY = ['BPXSY1', 'BPXSY2', 'BPXSY3', 'BPXSY4']
BP_DI = ['BPXDI1', 'BPXDI2', 'BPXDI3', 'BPXDI4']

total_list = NUMERIC+INQ+DPQ+DEMO
print(df_total.isnull().sum())
df_HR = df_total[total_list].dropna()
df_BSY = df_total[BP_SY]
df_BDI = df_total[BP_DI]

df_BSY = pd.DataFrame(np.where(df_BSY < 1, 0, df_BSY))
df_BDI = pd.DataFrame(np.where(df_BDI < 1, 0, df_BDI))

df_BSY = df_BSY.replace(0, np.nan)
df_BDI = df_BDI.replace(0, np.nan)

df_BSY = np.array(df_BSY.mean(axis=1))
df_BDI = np.array(df_BDI.mean(axis=1))

df_total = df_total[total_list]
df_total['BPXSY'] = df_BSY
df_total['BPXDI'] = df_BDI
print(df_total.isnull().sum())
df_total = df_total.dropna()

df_total[df_total[DPQ] < 1] = 0
df_total[df_total[DPQ] > 3] = np.nan

df_total = df_total.dropna()

df_total.to_csv(dir_name+'.csv')