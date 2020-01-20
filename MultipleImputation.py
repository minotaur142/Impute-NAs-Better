import random
import math
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
import random
import tqdm
from sklearn.preprocessing import LabelEncoder

def impute_missing_values(df,var_deviation_tolerance=0.97, actual_or_gaussian_residuals='actual', 
                          col_floor_ceiling_dict=None, scores=False):
    
    '''Impute missing values while minimizing distortion of variable distribution
    by creating a bagged model using other variables and adding residuals to output values
    
    Parameters:
    df: dataframe with missing values
    var_deviation_tolerance: target percent deviation from original variable distributions
    actual_or_guassian_residuals: apply residuals to model outputs from actual distribution or from
        a gaussian distribution based on residuals' means and variances
    col_floor_ceiling_dict: a dictionary with the variable name and a tuple of the min and max for variables 
        with a finite range. Use float(inf) or float(-inf) for variables that are limited in only one direction
    scores: return accuracy score of models per variable
    
    Returns:
    df: df with imputed values
    problems: columns that failed to impute
    column_scores: accuracy scores of imputation model on non-missing values
    '''
    df = df.copy()
    columns = df.columns
    type_dict = df.dtypes.to_dict()
    missing_columns = list(df.isna().sum()[df.isna().sum()>0].sort_values().index)
    have_columns = [i for i in columns if i not in missing_columns]
    column_scores = {}
    problems = []
    for col in tqdm.tqdm(missing_columns):
        try:
            percent_missing = df[col].isna().sum()/df.shape[0]
            m = math.ceil(percent_missing/((1/.97)-1))
            other_columns = [i for i in columns if i != col]
            na_index = df[df[col].isna()==1].index
            have_index = [i for i in df.index if i not in na_index]
            na_have_cols = set(df.loc[na_index,other_columns].dropna(axis=1).columns)
            have_have_cols = set(df.loc[have_index,other_columns].dropna(axis=1).columns)
            both_cols = na_have_cols.intersection(have_have_cols)
            int_df = pd.get_dummies(df.loc[:,both_cols],drop_first=True)
            X_have = int_df.loc[have_index,:]
            y_have = df[col][have_index]
            X_na = int_df.loc[na_index,:]
            if type_dict[col]=='object':
                le = LabelEncoder()
                y_have = le.fit_transform(y_have)
                df[col][have_index] = y_have
                rf = RandomForestClassifier()
                bagc = BaggingClassifier(base_estimator=rf,n_estimators=m)
                bagc.fit(X_have,y_have)
                column_scores[col]=bagc.score(X_have,y_have)
                resid_preds = bagc.predict(X_have)
                residuals = y_have-resid_preds
                preds = bagc.predict(X_na)
            else:
                bagr = BaggingRegressor(n_estimators=m)
                bagr.fit(X_have,y_have)
                column_scores[col] = bagr.score(X_have,y_have)
                resid_preds = bagr.predict(X_have)
                residuals = y_have-resid_preds
                preds = bagr.predict(X_na)
            if actual_or_gaussian_residuals=='actual':
                rand_resids = np.random.choice(residuals,len(X_na),replace=True)
            else:
                rand_resids = np.random.normal(residuals.mean(),residuals.std(),len(X_na))
            preds = preds + rand_resids
            if type_dict[col]=='object':
                preds = preds.round()
            if col_floor_ceiling_dict!=None:
                if col in col_floor_ceiling_dict.keys():
                        preds = np.clip(preds,col_floor_ceiling_dict[col][0],col_floor_ceiling_dict[col][1])
            df[col][na_index] = preds
            have_columns.append(col)
        except:
            problems.append(col)
    if scores == False:
        return df,problems
    else:
        return df, problems, column_scores
