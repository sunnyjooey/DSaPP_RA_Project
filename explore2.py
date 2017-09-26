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
        - quantile: quantile cutoff (decimal) for each feature, particularly useful for very skewed data 
        	put 1 to use all data
    Output:
        - a histogram for each feature 
    '''
    for feature in features_list:
        cutoff = df[feature].quantile(quantile)        
        sub_df = df[df[feature] < cutoff][feature]
        plt.figure()
        sub_df.plot.hist(title=feature)



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



def make_boxplots(df, y_var, features_list, figsize, quantile):
    '''
    Input: 
        - df: pandas dataframe
        - y_var: a continuous variable to be boxplotted
        - features_list: list of features to disaggregate each boxplot by
        - figsize: tuple of two numbers indicating figure size
        - quantile: quantile cutt-off (a decimal) for the data (y_var)
            particularly useful for very skewed data
            put 1 to use all data 
    Output:
        - a boxplot for each of the features 
    '''
    cutoff = df[y_var].quantile(quantile)        
    df = df[df[y_var]<cutoff]

    for feature in features_list:
        fig, ax = plt.subplots(figsize=figsize)
        sns.boxplot(x=feature, y=y_var, data=df,palette='rainbow')



def make_bar_box(df, features_list, figsize, y_var, quantile):
    '''
    Acombination of make_bargraphs and make_boxplots functions:
    	Makes one bar graph and one boxplot for each feature
    '''
    cutoff = df[y_var].quantile(quantile)        
    df = df[df[y_var]<cutoff]

    for feature in features_list:
        fig, ax = plt.subplots(figsize=figsize)
        sns.countplot(x=feature, data=df)
  
        fig1, ax1 = plt.subplots(figsize=figsize)
        sns.boxplot(x=feature, y=y_var, data=df,palette='rainbow')
