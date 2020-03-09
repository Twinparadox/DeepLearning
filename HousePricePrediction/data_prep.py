def makeSlidingWindows(origin_train, origin_test, train, test, lookBack=2, multiVariate=False, gtruthName='SPI'):

    if not multiVariate:
        train_df = pd.DataFrame(train[:,0], index=origin_train.index)
        test_df = pd.DataFrame(test[:,0], index=origin_test.index)
        train_df.columns = origin_train.columns
        test_df.columns = origin_test.columns
        column_list = list(train_df)
        
        for s in range(1, lookBack + 1):
            tmp_train = train_df[column_list].shift(s)
            tmp_test = test_df[column_list].shift(s)

            tmp_train.columns = "shift_" + tmp_train.columns + "_" + str(s)
            tmp_test.columns = "shift_" + tmp_test.columns + "_" + str(s)

            train_df[tmp_train.columns] = train_df[column_list].shift(s)
            test_df[tmp_test.columns] = test_df[column_list].shift(s)

        return train_df, test_df

    else:
        train_df = pd.DataFrame(train, index=origin_train.index)
        test_df = pd.DataFrame(test, index=origin_test.index)
        train_df.columns = origin_train.columns
        test_df.columns = origin_test.columns
        column_list = list(train_df)

        for s in range(1, lookBack + 1):
            tmp_train = train_df[column_list].shift(s)
            tmp_test = test_df[column_list].shift(s)

            tmp_train.columns = "shift_" + tmp_train.columns + "_" + str(s)
            tmp_test.columns = "shift_" + tmp_test.columns + "_" + str(s)

            train_df[tmp_train.columns] = train_df[column_list].shift(s)
            test_df[tmp_test.columns] = test_df[column_list].shift(s)  

if __name__ == "__main__":
    lookBack = 1
    multiVariate = True
    n_features = 1
    train_period1 = 128
    test_period1 = 12
    train_period2 = 32
    test_period2 = 12
    
    epochs = 100
    n_units = [20, 50, 100, 150, 200, 250, 300]
    

    df = pd.read_csv('./data/houseprice.csv')
    data = df[['SPI', 'CPI', 'M2', 'MMI', 'CBD']]
    column_list = list(data)

    train1, test1, train1_sc, test1_sc = splitTrainTest(data=data,
                                                        train_period=train_period1,
                                                        test_period=test_period1,
                                                        multiVariate=multiVariate,
                                                        period=1)
    train1_df, test1_df = makeSlidingWindows(origin_train=train1,
                                             origin_test=test1,
                                             train=train1_sc,
                                             test=test1_sc,
                                             lookBack=lookBack,
                                             multiVariate=multiVariate)