# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 18:52:37 2020

@author: nww73
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    df_train = pd.read_csv("data/train.csv", engine='c')
    df_test = pd.read_csv("data/test.csv", engine='c')
    
    print("train shape: ", df_train.shape)
    print("test shape: ", df_test.shape)
    
    print("df_train is null: ", df_train.isnull().sum().sum())
    print("df_test is null: ", df_test.isnull().sum().sum())
    
    df_train.target.value_counts(normalize=True)
    
    f,ax=plt.subplots(1,2, figsize=(12,4))
    df_train.target.value_counts().plot.pie(explode=[0,0.12],autopct='%1.3f%%',ax=ax[0])
    sns.countplot('target',data=df_train)
    plt.show()
    
    feat = df_train.columns.values[2:202]
    print("[features]")
    print(feat)
    
    # row distribution
    plt.figure(figsize=(15,5))
    sns.distplot(df_train[feat].mean(axis=1),color="green", kde=True,bins=100, label='train')
    sns.distplot(df_test[feat].mean(axis=1),color="red", kde=True,bins=100, label='test')
    plt.title("Distribution of mean values per row in the train and test dataset")
    plt.legend()
    plt.show()
    
    plt.figure(figsize=(15,5))
    sns.distplot(df_train[feat].std(axis=1),color="green", kde=True,bins=100, label='train')
    sns.distplot(df_test[feat].std(axis=1),color="red", kde=True,bins=100, label='test')
    plt.title("Distribution of Standard Deviation values per row in the train and test dataset")
    plt.legend()
    plt.show()
    
    # column distribution
    plt.figure(figsize=(15,5))
    plt.title("Distribution of mean values per column in the train and test set")
    sns.distplot(df_train[feat].mean(axis=0),color="blue",kde=True,bins=100, label='train')
    sns.distplot(df_test[feat].mean(axis=0),color="red", kde=True,bins=100, label='test')
    plt.legend()
    plt.show()
    
    plt.figure(figsize=(15,5))
    sns.distplot(df_train[feat].std(axis=0),color="blue", kde=True,bins=100, label='train')
    sns.distplot(df_test[feat].std(axis=0),color="red", kde=True,bins=100, label='test')
    plt.title("Distribution of Standard Deviation values per columns in the train and test dataset")
    plt.legend()
    plt.show()
    
    
    df_train.loc[df_train.target == 0][feat].mean(axis=1)
    df_train.loc[df_train.target == 1][feat].mean(axis=1)
    plt.figure(figsize=(15,5))
    sns.distplot(df_train.loc[df_train.target == 0][feat].mean(axis=1),color="red", kde=True,bins=100,label='target = 0')
    sns.distplot(df_train.loc[df_train.target == 1][feat].mean(axis=1),color="blue", kde=True,bins=100,label='target = 1')
    plt.title("Distribution of mean values per row in the train set grouped by Target")
    plt.legend()
    plt.show()
    
    df_train.loc[df_train.target == 0][feat].mean()
    df_train.loc[df_train.target == 1][feat].mean()
    plt.figure(figsize=(15,5))
    sns.distplot(df_train.loc[df_train.target == 0][feat].mean(),color="red", kde=True,bins=100,label='target = 0')
    sns.distplot(df_train.loc[df_train.target == 1][feat].mean(),color="green", kde=True,bins=100,label='target = 1')
    plt.title("Distribution of mean values per column in the train set grouped by Target")
    plt.legend()
    plt.show()
    
    
    # Correaltion
    df_train.corr()
    train_cor = df_train.drop(["target"], axis=1).corr()
    train_cor = train_cor.values.flatten()
    train_cor = train_cor[train_cor != 1]
    plt.figure(figsize=(15,5))
    sns.distplot(train_cor)
    plt.xlabel("Correlation values found in train excluding 1")
    plt.ylabel("Density")
    plt.title("Correlation between features")
    plt.show()
    
    
    df_test.corr()
    test_cor = df_test.corr()
    test_cor = test_cor.values.flatten()
    test_cor = test_cor[test_cor != 1]
    plt.figure(figsize=(15,5))
    sns.distplot(test_cor)
    plt.xlabel("Correlation values found in test excluding 1")
    plt.ylabel("Density")
    plt.title("Correlation between features")
    plt.show()
