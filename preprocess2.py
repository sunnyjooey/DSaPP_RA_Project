import pandas as pd


def fill_nulls(df, variable, fill_value):
    '''
    Input:
        - df: pandas dataframe
        - variable: variable with missing values to fill
        - fill_value: the value to fill the nulls
    Output: the dataframe with the values filled
    '''
    if fill_value == int or fill_value == float:
        df[variable] = df[variable].fillna(fill_value)
    elif fill_value == 'median':
        df[variable] = df[variable].fillna(df[variable].median())
    elif fill_value == 'mean':
        df[variable] = df[variable].fillna(df[variable].mean())
    elif fill_value == 'mode':
        df[variable] = df[variable].fillna(df[variable].mode()[0])
    else:
        print('This is not a valid fill value. Nothing was performed on the data')
    return df



def add_dummy(df, variable_list, drop_one=False, drop_original=False):
    '''
    Input: 
        - df: pandas dataframe
        - variable_list: a list of variables to dummitize
        - drop_one: whether to drop first dummy
        - drop_original: whether to drop original categorical variable
    Output: dataframe with tht dummy variables added
    '''
    for variable in variable_list:
            
        df_dummy = pd.get_dummies(df[variable], drop_first=drop_one)
        df = pd.concat([df, df_dummy], axis=1)
        if drop_original:
            df = df.drop(variable, 1)
    return df



def transform(X_train, X_test):
    '''
    Standardize continuous variable values
    Input: X_train and X_test dataframes
    Output: Standardized dataframes
    '''
    variables = X_train.columns.tolist()
    to_standardize = []
    for variable in variables:
        if set(X_train[variable].unique().tolist()) != {0, 1}:
            to_standardize.append(variable)
            
    for var in to_standardize:
        scalar = preprocessing.RobustScaler()
        X_train[var] = scalar.fit_transform(X_train[var])
        
        #Since scalar object was fitted on X_train, X_test will be transformed on the same scale
        X_test[var] = scalar.transform(X_test[var])
    
    return X_train, X_test

