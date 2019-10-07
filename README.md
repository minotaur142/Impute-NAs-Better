# Impute-NAs-Better
Impute missing values while minimizing distortion of overall variable distributions by:

1. Using available columns per row to create a bagged model
2. Applying that model to non-NA rows to find distribution of residuals
3. Imputing values by adding a random residual from to the model's output

This imputer is an implementation of a technique described in the following paper:

Joseph L. Schafer & Maren K. Olsen (1998) Multiple Imputation for Multivariate Missing-Data Problems: A Data Analyst's Perspective, Multivariate Behavioral Research, 33:4, 545-571, DOI: 10.1207/s15327906mbr3304_5
