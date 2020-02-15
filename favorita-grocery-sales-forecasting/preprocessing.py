# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 18:09:25 2020

@author: nww73
"""
import os
import pandas as pd
import numpy as np
import multiprocessing
import datetime
import matplotlib.pyplot as plt
from multiprocessing import Pool
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample
from dateutil.relativedelta import relativedelta

def preprocess_train(df_train, df_oil, df_stores, df_holidays_events, df_items, df_cpi):
    df_train = df_train.dropna()
    
    df_train['date'] = pd.to_datetime(df_train['date'], format='%Y-%m-%d')
    df_train = df_train[df_train['date'] > '2014-12-31']
    df_train['onpromotion'] = pd.to_numeric(df_train['onpromotion'])
    df_train = df_train[df_train['unit_sales'] >= 0]
    
    mask = df_train['id'].isin([93189596,76939364,77960441,76693277,77659454])
    df_train = df_train[~mask]
    print(df_train.describe())
    
    # is_salary
    # 16일과 마지막날은 salary True
    df_train['is_salary'] = (df_train['date'].dt.day == 15) | (df_train['date'].dt.is_month_end)    
    df_train = df_train.sort_values(['store_nbr','date'], ascending=[True, True])
    
    # Add is_salary_last
    df_train['is_salary_last'] = np.where(df_train['is_salary'],0,
            np.where(df_train['date'].dt.day < 15, df_train['date'].dt.day, df_train['date'].dt.day-15))
    
    # Add oil price
    # oil 데이터에는 결측치가 존재함
    # 거래가 없는 날은 이전 날의 유가를 기준으로 하는 결측치 보완 방식 적용 
    df_oil = df_oil.rename(columns={'Date':'date_lag', 'Price':'wti_price'})
    df_oil['date_lag'] = pd.to_datetime(df_oil['date_lag'])
    df_oil = df_oil.set_index('date_lag')
    df_train['date_lag'] = df_train['date']+pd.DateOffset(months=6)    
    df_train = pd.merge(df_train, df_oil, left_on=['date_lag'], right_index=True, how='left')
    df_train['wti_price'] = df_train['wti_price'].fillna(method='ffill') 
    del df_train['date_lag']
    print(df_train.describe())
    
    # Add cpi
    df_train['year'] = pd.to_numeric(df_train['date'].dt.year)    
    df_cpi = df_cpi.rename(columns={'YEAR':'year', 'CPI':'cpi'})
    df_cpi = df_cpi.set_index('year')    
    df_train = pd.merge(df_train, df_cpi, left_on=['year'], right_index=True, how='left')
    del df_train['year']
    print(df_train.describe())
    
    # Add Perishable
    df_items = df_items[['item_nbr','perishable']]
    df_items = df_items.set_index('item_nbr')    
    df_train = pd.merge(df_train, df_items, left_on=['item_nbr'], right_index=True, how='left')
    print(df_train.describe())
    
    # Add is_holiday
    df_train = pd.merge(df_train, df_holidays_events, left_on=['date'], right_index=True, how='left')
    df_train['is_holidays'] = df_train['is_holidays'].fillna(0)
    df_train.dropna()
    
    df_train['is_salary'] = np.where(df_train['is_salary'], 1, 0)
    df_train['onpromotion'] = np.where(df_train['onpromotion'], 1, 0)
    print(df_train.describe())
    
    return df_train
    
def preprocess_test(df_test, df_oil, df_stores, df_holidays_events, df_items, df_cpi):
    df_test['onpromotion'] = pd.to_numeric(df_test['onpromotion'])
    
    df_test['date'] = pd.to_datetime(df_test['date'], format='%Y-%m-%d')
    
    # is_salary
    # 15일과 마지막날은 salary True
    df_test['is_salary'] = (df_test['date'].dt.day == 15) | (df_test['date'].dt.is_month_end)    
    df_test = df_test.sort_values(['store_nbr','date'], ascending=[True, True])
    
    # Add is_salary_last
    df_test['is_salary_last'] = np.where(df_test['is_salary'],0,
            np.where(df_test['date'].dt.day < 15, df_test['date'].dt.day, df_test['date'].dt.day-15))
    
    # Add oil price
    # oil 데이터에는 결측치가 존재함
    # 거래가 없는 날은 이전 날의 유가를 기준으로 하는 결측치 보완 방식 적용 
    df_oil = df_oil.rename(columns={'Date':'date_lag', 'Price':'wti_price'})
    df_oil['date_lag'] = pd.to_datetime(df_oil['date_lag'])
    df_oil = df_oil.set_index('date_lag')
    df_test['date_lag'] = df_test['date']+pd.DateOffset(months=6)    
    df_test = pd.merge(df_test, df_oil, left_on=['date_lag'], right_index=True, how='left')
    df_test['wti_price'] = df_test['wti_price'].fillna(method='ffill')
    del df_test['date_lag']
    
    # Add cpi
    df_test['year'] = pd.to_numeric(df_test['date'].dt.year)    
    df_cpi = df_cpi.rename(columns={'YEAR':'year', 'CPI':'cpi'})
    df_cpi = df_cpi.set_index('year')    
    df_test = pd.merge(df_test, df_cpi, left_on=['year'], right_index=True, how='left')
    del df_test['year']
    
    # Add Perishable
    df_items = df_items[['item_nbr','perishable']]
    df_items = df_items.set_index('item_nbr')    
    df_test = pd.merge(df_test, df_items, left_on=['item_nbr'], right_index=True, how='left')
    
    # Add is_holiday
    df_test = pd.merge(df_test, df_holidays_events, left_on=['date'], right_index=True, how='left')
    print(df_test.isnull().sum())
    df_test['is_holidays'] = df_test['is_holidays'].fillna(0)
    df_test.dropna()    
    
    df_test['is_salary'] = np.where(df_test['is_salary'], 1, 0)
    df_test['onpromotion'] = np.where(df_test['onpromotion'], 1, 0)
    
    return df_test

if __name__ == '__main__':
    num_cores = 6

    df_train = pd.read_csv('./data/train.csv', engine='c', low_memory=False)
    df_test = pd.read_csv('./data/test.csv', engine='c', low_memory=False)
    df_oil = pd.read_csv('./data/oil.csv', engine='c', low_memory=False)
    df_stores = pd.read_csv('./data/stores.csv', engine='c', low_memory=False)
    df_holidays_events = pd.read_csv('./data/holidays_events.csv', engine='c', low_memory=False)
    df_items = pd.read_csv('./data/items.csv', engine='c', low_memory=False)
    df_cpi = pd.read_csv('./data/cpi.csv', engine='c', low_memory=False)

    # 실제 휴일만 고려
    df_holidays_events = df_holidays_events[df_holidays_events['transferred'] == False]
    df_holidays_events = df_holidays_events[df_holidays_events['type'] != 'Work Day']
    df_holidays_events['is_holidays'] = 1
    df_holidays_events['date'] = pd.to_datetime(df_holidays_events['date'])
    df_holidays_events = df_holidays_events[['date','is_holidays']]
    df_holidays_events = df_holidays_events.drop_duplicates('date', keep='first')
    df_holidays_events = df_holidays_events.set_index('date')

    df_made_train = preprocess_train(df_train, df_oil, df_stores, df_holidays_events, df_items, df_cpi)
    df_made_train = df_made_train.set_index('id')
    df_made_test = preprocess_test(df_test, df_oil, df_stores, df_holidays_events, df_items, df_cpi)
    df_made_test = df_made_test.set_index('id')

    df_made_train.to_csv('./data/prep_train.csv', sep=',', na_rep='NaN')
    df_made_test.to_csv('./data/prep_test.csv', sep=',', na_rep='NaN')