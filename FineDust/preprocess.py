# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 14:09:47 2020

@author: nww73
"""
import pandas as pd
import numpy as np
import gc

import matplotlib.pyplot as plt
#import seaborn as sns

def outliers_iqr(data):
    q1, q3 = np.percentile(data, [25,75])
    iqr = q3-q1
    lower_bound = q1-(iqr*1.5)
    upper_bound = q3+(iqr*1.5)
    return np.where((data>upper_bound) | (data<lower_bound))

def outliers_z_score(data, threshold=3):
    mean = np.mean(data)
    std = np.std(data)
    z_scores = [(y-mean)/std for y in data]
    return np.where(np.abs(z_scores)>threshold)

def remove_outlier(data):
    value = ['PM10', 'NO2', 'CO', 'SO2', 'avg_tmp', 'min_tmp', 'max_tmp',
             'precipitation', 'max_inst_wind', 'max_inst_wind_direct',
             'max_avg_wind_direct', 'avg_wind', 'min_humid', 'avg_humid',
             'avg_hPa', 'avg_total_cloud', 'avg_mid_cloud', 'avg_gtmp',
             'avg_gtmp5', 'avg_gtmp10', 'avg_gtmp150', 'avg_gtmp300']

if __name__ == '__main__':
    
    # air
    air_data_type = {'날짜':'str',
                     '측정소명':'str',
                     '미세먼지':'float16',
                     '초미세먼지':'float16',
                     '오존':'float16',
                     '이산화질소':'float16',
                     '일산화탄소':'float16',
                     '아황산가스':'float16'}
    
    df_2010 = pd.read_csv('data/air_2010.csv', engine='c', dtype=air_data_type,
                          parse_dates=['날짜'])
    df_2011 = pd.read_csv('data/air_2011.csv', engine='c', dtype=air_data_type,
                          parse_dates=['날짜'])
    df_2012 = pd.read_csv('data/air_2012.csv', engine='c', dtype=air_data_type,
                          parse_dates=['날짜'])
    df_2013 = pd.read_csv('data/air_2013.csv', engine='c', dtype=air_data_type,
                          parse_dates=['날짜'])
    df_2014 = pd.read_csv('data/air_2014.csv', engine='c', dtype=air_data_type,
                          parse_dates=['날짜'])
    df_2015 = pd.read_csv('data/air_2015.csv', engine='c', dtype=air_data_type,
                          parse_dates=['날짜'])
    df_2016 = pd.read_csv('data/air_2016.csv', engine='c', dtype=air_data_type,
                          parse_dates=['날짜'])
    df_2017 = pd.read_csv('data/air_2017.csv', engine='c', dtype=air_data_type,
                          parse_dates=['날짜'])
    
    df_air = pd.concat([df_2010, df_2011, df_2012, df_2013, df_2014, df_2015, df_2016, df_2017])
    df_air.columns = ['date','locale','PM10','PM2.5','O3','NO2','CO','SO2']
    df_air['date'] = df_air['date'].dt.strftime('%Y-%m-%d')
    del df_air['PM2.5']
    del [df_2010, df_2011, df_2012, df_2013, df_2014, df_2015, df_2016, df_2017]
    gc.collect()
    
    mask = df_air['locale'].isin(['평균'])
    df_air = df_air[~mask]
    del mask
    del df_air['O3']
    gc.collect()
    
    print(df_air.dtypes)    
    
    # climate
    climate_data_type = {'id':'uint8',
                         'locale':'str',
                         'date':'str',
                         'avg_tmp':'float16',
                         'min_tmp':'float16',
                         'max_tmp':'float16',
                         'precipitation':'float16',
                         'max_inst_wind':'float16',
                         'max_inst_wind_direct':'float16',
                         'max_avg_wind_direct':'float16',
                         'avg_wind':'float16',
                         'min_humid':'float16',
                         'avg_humid':'float16',
                         'avg_hPa':'float16',
                         'avg_total_cloud':'float16',
                         'avg_mid_cloud':'float16',
                         'avg_gtmp':'float16',
                         'avg_gtmp5':'float16',
                         'avg_gtmp10':'float16',
                         'avg_gtmp150':'float16',
                         'avg_gtmp300':'float16'}
    
    df_climate = pd.read_csv('data/climate.csv', engine='c', dtype=climate_data_type,
                             parse_dates=['date'])
    df_climate['date'] = df_climate['date'].dt.strftime('%Y-%m-%d')
    del df_climate['id']
    del df_climate['locale']
    gc.collect()    
    
    df_air = df_air.set_index('date')
    df_climate = df_climate.set_index('date')
    df_result = pd.merge(df_air, df_climate, left_on=['date'], right_index=True, how='left')
    df_result = df_result.reset_index()
    df_result = df_result.sort_values(by=['date'],axis=0)
    
    print(df_result.dtypes)
    df_result = df_result.fillna(0)

    remove_outlier(df_result)
    df_result.to_csv('./data/prep_data.csv', sep=',', na_rep='NaN', encoding='utf-8-sig')

