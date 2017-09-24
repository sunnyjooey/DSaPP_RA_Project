import pandas as pd
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
import seaborn as sns


def basic_data_description(df):
    '''
    Input: pandas dataframe
    Output: table of the basic statistics, median values of all continuous variables    
    '''
    desc = df.describe().round(decimals=2)
    medians = df.median().round(decimals=2).to_frame().rename(columns={0: "median"}).T
    table = pd.concat([desc, medians], axis=0)
    return table


def make_histograms(df, features_list, quantile):
    '''
    Input: 
        - df: pandas dataframe
        - features_list: list of features
        - quantile: desired quatile cutoff (decimal), particularly useful for very skewed data 
            if no quatile cutoff desired, put 1 
    Output:
        - two histograms for each feature (above and below the quantile cutoff)
    '''
    for feature in features_list:
        cutoff = df[feature].quantile(quantile)        
        top_sub_df = df[df[feature] >= cutoff][feature]
        bot_sub_df = df[df[feature] < cutoff][feature]
        
        plt.figure()
        top_sub_df.plot.hist(title=feature + ' top ' + str(quantile))
        plt.figure()
        bot_sub_df.plot.hist(title=feature + ' bottom ' + str(1-quantile))


def make_bargraphs(df, features_list, figsize):
    '''
    Input: 
        - df: pandas dataframe
        - features_list: list of features
        - figsize: tuple of two numbers indicating figure size
    Output:
        - a bar graph of category counts for each feature
    '''
    for feature in features_list:
        fig, ax = plt.subplots(figsize=figsize)
        sns.countplot(x=feature, data=df)


def make_boxplots(df, y_var, features_list, figsize, quantile=None):
    '''
    Input: 
        - df: pandas dataframe
        - y_var: a continuous variable to be boxplotted
        - features_list: list of features to disaggregate each boxplot by
        - figsize: tuple of two numbers indicating figure size
        - quantile: optional quantile cutt-off (a decimal) for the data (y_var)
            particularly useful for very skewed data 
    Output:
        - a boxplot for each of the features 
    '''
    cutoff = df[y_var].quantile(quantile)        
    df = df[df[y_var]<cutoff]
    for feature in features_list:
        fig, ax = plt.subplots(figsize=figsize)
        sns.boxplot(x=feature, y=y_var, data=df,palette='rainbow')