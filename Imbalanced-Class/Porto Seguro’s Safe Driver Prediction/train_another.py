import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score


def describe_missing_values(df):
    na_percent = {}
    N = df.shape[0]
    for column in df:
        na_percent[column] = df[column].isnull().sum() * 100 / N

    na_percent = dict(filter(lambda x: x[1] != 0, na_percent.items()))
    plt.bar(range(len(na_percent)), na_percent.values())
    plt.ylabel('Percent')
    plt.xticks(range(len(na_percent)), na_percent.keys(), rotation='vertical')
    plt.show()
    
def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))

def gini(actual, pred, cmpcol = 0, sortcol = 1):
    assert( len(actual) == len(pred) )
    all = np.asarray(np.c_[ actual, pred, np.arange(len(actual)) ], dtype=np.float)
    all = all[ np.lexsort((all[:,2], -1*all[:,1])) ]
    totalLosses = all[:,0].sum()
    giniSum = all[:,0].cumsum().sum() / totalLosses
    
    giniSum -= (len(actual) + 1) / 2.
    return giniSum / len(actual)
 
def gini_normalized(a, p):
    return gini(a, p) / gini(a, a)

def gini_xgb(preds, dtrain):
    labels = dtrain.get_label()
    gini_score = gini_normalized(labels, preds)
    return 'gini', gini_score

def resampling(df_train, ratio):    
    train_freq = df_train['target'].value_counts()    
    print(train_freq)
    train_freq_mean = train_freq[0]
    
    # Under & Over Sampling store_nbr
    df_list = []
    target_max = 2
    multiple = ratio
    
    for i in range(0, target_max):
        df_list.append(df_train[df_train['target']==i])
    
    for i in range(0, target_max):
        if i==0:
            df_list[i] = df_list[i].sample(n=int(train_freq_mean*multiple), random_state=123, replace=True)
        else:
            df_list[i] = df_list[i].sample(n=train_freq_mean, random_state=123, replace=True)
        
    df_sampling_train = pd.concat(df_list)
    train_freq = df_sampling_train['target'].value_counts()
    
    return pd.DataFrame(df_sampling_train)

def XGBRandom(train, test):
    kfold = 3
    param_comb = 20
    skf = StratifiedKFold(n_splits=kfold, random_state=42, shuffle=True)
    
    params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.4, 0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 5, 7],
        'n_estimator':[100, 300, 500],
        'learning_rate':[0.01, 0.02, 0.03]
    }
    
    xgb = XGBClassifier(learning_rate = 0.01, n_estimator=300,
                        objective='binary:logistic', silent=True, nthread=1,
                        tree_method='gpu_hist', predictor='gpu_predictor',
                        num_boost_round=500, early_stopping_rounds=70)
    
    X = train.drop(['id', 'target'], axis=1)
    y = train.target.values
    test_df = test
    test = test.drop('id', axis=1)
    
    random_search = RandomizedSearchCV(xgb, param_distributions=params,
                                       n_iter=param_comb, scoring='roc_auc',
                                       n_jobs=1, cv=skf.split(X,y),
                                       verbose=1, random_state=1001)
    # Here we go
    start_time = timer(None) # timing starts from this point for "start_time" variable
    random_search.fit(X, y)
    timer(start_time) # timing ends here for "start_time" variable
    
    print('\n All results:')
    print(random_search.cv_results_)
    print('\n Best estimator:')
    print(random_search.best_estimator_)
    print('\n Best normalized gini score for %d-fold search with %d parameter combinations:' % (kfold, param_comb))
    print(random_search.best_score_ * 2 - 1)
    print('\n Best hyperparameters:')
    print(random_search.best_params_)
    results = pd.DataFrame(random_search.cv_results_)
    results.to_csv('xgb-random-grid-search-results-01.csv', index=False)
    
    y_test = random_search.predict_proba(test)
    results_df = pd.DataFrame(data={'id':test_df['id'], 'target':y_test[:,1]})
    results_df.to_csv('submission-random-grid-search-xgb-porto-01.csv', index=False)
    
    pred = random_search.best_estimator_.predict(X)
    
    print(confusion_matrix(y, pred))
    print(classification_report(y, pred, labels=[0, 1]))
    
def XGBGrid(train, test):
    kfold = 3
    param_comb = 7
    skf = StratifiedKFold(n_splits=kfold, random_state=42, shuffle=True)
    
    params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.4, 0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 5, 7],
        'n_estimator':[100, 300, 500],
        'learning_rate':[0.01, 0.02, 0.03]
    }
    
    xgb = XGBClassifier(learning_rate = 0.01, n_estimator=300,
                        objective='binary:logistic', silent=True, nthread=1,
                        tree_method='gpu_hist', predictor='gpu_predictor',
                        num_boost_round=500, early_stopping_rounds=70)
    
    X = train.drop(['id', 'target'], axis=1)
    y = train.target.values
    test_df = test
    test = test.drop('id', axis=1)
    
    random_search = GridSearchCV(xgb, param_grid=params, scoring='roc_auc',
                                 n_jobs=1, cv=skf.split(X,y), verbose=1)
    # Here we go
    start_time = timer(None) # timing starts from this point for "start_time" variable
    random_search.fit(X, y)
    timer(start_time) # timing ends here for "start_time" variable
    
    print('\n All results:')
    print(random_search.cv_results_)
    print('\n Best estimator:')
    print(random_search.best_estimator_)
    print('\n Best normalized gini score for %d-fold search with %d parameter combinations:' % (kfold, param_comb))
    print(random_search.best_score_ * 2 - 1)
    print('\n Best hyperparameters:')
    print(random_search.best_params_)
    results = pd.DataFrame(random_search.cv_results_)
    results.to_csv('xgb-random-grid-search-results-01.csv', index=False)
    
    y_test = random_search.predict_proba(test)
    results_df = pd.DataFrame(data={'id':test_df['id'], 'target':y_test[:,1]})
    results_df.to_csv('submission-random-grid-search-xgb-porto-01.csv', index=False)
    
def XGBBest(train, test):
    kfold = 5
    skf = StratifiedKFold(n_splits=kfold, random_state=42)
    params = {
        'min_child_weight': 10.0,
        'objective': 'binary:logistic',
        'max_depth': 7,
        'max_delta_step': 1.8,
        'colsample_bytree': 0.4,
        'subsample': 0.8,
        'eta': 0.025,
        'gamma': 0.65,
        'num_boost_round' : 700
    }
    
    X = train.drop(['id', 'target'], axis=1).values
    y = train.target.values
    test_id = test.id.values
    test = test.drop('id', axis=1)
    
    sub = pd.DataFrame()
    sub['id'] = test_id
    sub['target'] = np.zeros_like(test_id)
    
    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        print('[Fold %d/%d]' % (i + 1, kfold))
        X_train, X_valid = X[train_index], X[test_index]
        y_train, y_valid = y[train_index], y[test_index]
        # Convert our data into XGBoost format
        d_train = xgb.DMatrix(X_train, y_train)
        d_valid = xgb.DMatrix(X_valid, y_valid)
        d_test = xgb.DMatrix(test.values)
        watchlist = [(d_train, 'train'), (d_valid, 'valid')]
        
        # Train the model! We pass in a max of 1,600 rounds (with early stopping after 70)
        # and the custom metric (maximize=True tells xgb that higher metric is better)
        mdl = xgb.train(params, d_train, 1600, watchlist, early_stopping_rounds=70,
                        feval=gini_xgb, maximize=True, verbose_eval=100)
        
        print('[Fold %d/%d Prediciton:]' % (i + 1, kfold))
        # Predict on our test data
        p_test = mdl.predict(d_test, ntree_limit=mdl.best_ntree_limit)
        sub['target'] += p_test/kfold

    sub.to_csv('StratifiedKFold.csv', index=False)
    

if __name__ == '__main__':
    train=pd.read_csv('data/train.csv', engine='c', na_values=-1)
    test=pd.read_csv('data/test.csv', engine='c', na_values=-1)
    train.isnull().values.any()
    
    features = train.drop(['id','target'], axis=1).values
    targets = train.target.values
    
    ax = sns.countplot(x = targets ,palette="Set2")
    sns.set(font_scale=1.5)
    ax.set_xlabel(' ')
    ax.set_ylabel(' ')
    fig = plt.gcf()
    fig.set_size_inches(10,5)
    ax.set_ylim(top=700000)
    for p in ax.patches:
        ax.annotate('{:.2f}%'.format(100*p.get_height()/len(targets)), (p.get_x()+ 0.3, p.get_height()+10000))
    
    plt.title('Distribution of 595212 Targets')
    plt.xlabel('Initiation of Auto Insurance Claim Next Year')
    plt.ylabel('Frequency [%]')
    plt.show()

    sns.set(style="white")

    # Compute the correlation matrix
    corr = train.corr()
    
    
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))
    
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    
    plt.show()
    
    unwanted = train.columns[train.columns.str.startswith('ps_calc_')]
    train = train.drop(unwanted, axis=1)  
    test = test.drop(unwanted, axis=1)  
    
    
    data=[]
    for f in train.columns:
        # Defining the role
        if f =='target':
            role='target'
        elif f == 'id':
            role = 'id'
        else:
            role = 'input'
            
        print(train[f].dtype)
        # Defining the level
        if 'bin' in f or f == 'target':
            level = 'binary'
        elif 'cat' in f or f == 'id':
            level = 'nominal'
        elif train[f].dtype == 'float64':
            level = 'interval'
        elif train[f].dtype == 'int64':
            level = 'ordinal'
        
        # Initialize keep to True for all variables except for id
        keep = True
        if f == 'id':
            keep = False
        
        # Defining the data type
        dtype = train[f].dtype
        
        # Creating a Dict that contains all the metadata for the variable
        f_dict = {
            'varname': f,
            'role':role,
            'level':level,
            'keep':keep,
            'dtype':dtype
        }
        data.append(f_dict)
    
    meta = pd.DataFrame(data, columns=['varname', 'role', 'level', 'keep', 'dtype'])
    meta = meta.set_index('varname')
    
    print(meta)
    
    pd.DataFrame({'count':meta.groupby(['role', 'level'])['role'].size()}).reset_index()
    
    v = meta[(meta.level == 'interval') & (meta.keep)].index
    train[v].describe()
    v = meta[(meta.level == 'ordinal') & (meta.keep)].index
    train[v].describe()
    v = meta[(meta.level=='binary') & (meta.keep)].index
    train[v].describe()
    
    # Check Missing Value   
    print("Missing values for Train dataset")
    describe_missing_values(train)
    
    print("Missing values for Test dataset")
    describe_missing_values(test)
    
    # NA Remove
    test_id = test['id']
    train = train.drop(["ps_car_03_cat", "ps_car_05_cat"], axis=1)
    test = test.drop(["ps_car_03_cat","ps_car_05_cat"], axis=1)
    
    # fill NA
    cat_cols = [col for col in train.columns if 'cat' in col]
    bin_cols = [col for col in train.columns if 'bin' in col]
    con_cols = [col for col in train.columns if col not in bin_cols + cat_cols]# 최빈값으로 대체
    for col in cat_cols:
        train[col] = train[col].fillna(value=train[col].mode()[0])
        test[col] = test[col].fillna(value=test[col].mode()[0])
        
    # 최빈값으로 대체
    for col in bin_cols:
        train[col] = train[col].fillna(value=train[col].mode()[0])
        test[col] = test[col].fillna(value=test[col].mode()[0])
        
    # 평균값으로 대체
    for col in con_cols:
        if col != 'id' and col != 'target':
            train[col] = train[col].fillna(value=train[col].mean())
            test[col] = test[col].fillna(value=test[col].mean())
    
    print("Missing values for Train dataset")
    describe_missing_values(train)
    
    print("Missing values for Test dataset")
    describe_missing_values(test)
    
    print(train.dtypes)
    
    selector = VarianceThreshold(threshold=.01)
    selector.fit(train.drop(["id", "target"], axis=1)) # Fit to train without id and target variables
    f = np.vectorize(lambda x:not x) # Function to toggle boolean array elemets
    v = train.drop(['id', 'target'], axis=1).columns[f(selector.get_support())]
    print('{} variables have too low variance.'.format(len(v)))
    print('These variables are {}'.format(list(v)))    

    '''
    train = train.drop(['ps_ind_10_bin', 'ps_ind_11_bin', 'ps_ind_12_bin',
                              'ps_ind_13_bin', 'ps_car_10_cat', 'ps_car_12', 'ps_car_14'], axis=1)
    test = test.drop(['ps_ind_10_bin', 'ps_ind_11_bin', 'ps_ind_12_bin',
                            'ps_ind_13_bin', 'ps_car_10_cat', 'ps_car_12', 'ps_car_14'], axis=1)
    '''
    
    XGBRandom(train, test)
    #XGBGrid(train,test)
    #XGBBest(train, test)