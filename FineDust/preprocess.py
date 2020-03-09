# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 14:09:47 2020

@author: nww73
"""
import pandas as pd
import numpy as np
import gc

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.formula.api import ols
import seaborn as sns

scaler = StandardScaler()

def outliers_iqr(data):
    data = data.astype('float32')
    mean = data.mean()
    q1, q3 = np.percentile(data, [25,75])
    iqr = q3-q1
    lower_bound = q1-(iqr*1.5)
    upper_bound = q3+(iqr*1.5)
    
    return np.where((data>upper_bound) | (data<lower_bound), mean, data)

def outliers_z_score(data, threshold=3):
    data = data.astype('float32')
    mean = data.mean()
    std = data.std()
    z_scores = [(y-mean)/std for y in data]
    
    print(data, mean)
    
    return np.where(np.abs(z_scores)>threshold, mean, data)

def remove_outlier(data):
    input_data = data.columns
    print(input_data)
    
    for cols in data:
        if cols!='PM10' and cols!='locale':
            data[cols] = outliers_iqr(data[cols])

    return data    
        
if __name__ == '__main__':
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    
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
    
    df_climate = pd.read_csv('data/climate2.csv', engine='c',
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
    df_result = df_result.set_index('date')
    
    print(df_result.dtypes)
    df_result = df_result.fillna(0)
    
    print(df_result['avg_humid'].describe())
    df_result = remove_outlier(df_result)
    
    #del [df_result['locale']]
       
    #df_result = df_result.drop(['avg_hPa_g', 'avg_hPa_o', 'min_hPa_o', 'max_hPa_o'], axis=1)
    
    '''
    ls = df_result.columns
    corrs = df_result.corr()
    print(corrs)
    
    hm = sns.heatmap(df_result.corr(),annot=True,cmap='RdYlGn',linewidths=0.2, 
                vmax=1, vmin=-1, fmt='3.1f', xticklabels=True, yticklabels=True)
    print(hm.get_ylim())
    bottom, top = hm.get_ylim()
    fig=plt.gcf()
    fig.set_size_inches(25,25)
    plt.ylim(bottom+1.0, top-0.5)
    plt.savefig('variance.png', dpi=300)
    
    
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    df_result = df_result.drop(['PM10',
                                'day_max_snow', 'day_max_new_snow', 'sun_time', 'sun_sum', 'max_wind',
                                'avg_hPa_g', 'avg_hPa_o', 'min_hPa_o', 'max_hPa_o',
                                'max_tmp', 'min_tmp',
                                'avg_gtmp5', 'avg_gtmp10', 'avg_gtmp50',
                                'avg_gtmp100', 'avg_gtmp300', 'avg_gtmp', 'min_chosang',
                                'sun_possible',
                                'avg_humid', 'avg_dew',
                                'avg_total_cloud',
                                'max_inst_wind', 'max_wind_direct',
                                'avg_wind'], axis=1)
    vif = pd.DataFrame()
    vif["VIF Factor"] = [variance_inflation_factor(
        df_result.values, i) for i in range(df_result.shape[1])]
    vif["features"] = df_result.columns
    print(vif)
    
    #sns.pairplot(df_result)
    
    df_result = scaler.fit_transform(df_result)
    pca = PCA()
    pca.fit(df_result)
    
    p = pca
    
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of components')
    plt.ylabel('Cumulative explained variance')
    plt.show()
    '''
    
    df_result = df_result.drop(['day_max_snow', 'day_max_new_snow', 'sun_time', 'sun_sum', 'max_wind',
                                'avg_hPa_g', 'avg_hPa_o', 'min_hPa_o', 'max_hPa_o',
                                'max_tmp', 'min_tmp',
                                'avg_gtmp5', 'avg_gtmp10', 'avg_gtmp50',
                                'avg_gtmp100', 'avg_gtmp300', 'avg_gtmp', 'min_chosang',
                                'sun_possible',
                                'avg_humid', 'avg_dew',
                                'avg_total_cloud',
                                'max_inst_wind', 'max_wind_direct',
                                'avg_wind'], axis=1)
    
    df_result.to_csv('./data/prep_data.csv', sep=',', na_rep='NaN', encoding='utf-8-sig')