import numpy as np
from sklearn.metrics import r2_score
from scipy.stats import pearsonr


def get_metrics(truth, preds):
    """
        Iterates over experiments and prints the mean r-squared value and Pearson-r value of those experiments.
    """
    experiments = truth["id"].unique()
    results_dictionary = {'Precursor [ug/m3]' : {}, 'Gas [ug/m3]' : {}, 'Aerosol [ug_m3]' : {}}
    for feature in results_dictionary:
        for exp in experiments:
            column = truth['id'] == exp
            results_dictionary[feature]['r2'] = []
            results_dictionary[feature]['Pearson'] = []
            results_dictionary[feature]['r2'].append(r2_score(truth[column][feature], preds[column][feature]))
            results_dictionary[feature]['Pearson'].append(pearsonr(truth[column][feature], preds[column][feature])[0])

    results = {}
    for feature in results_dictionary:
        R2 = np.mean(results_dictionary[feature]['r2'])
        Pearson = np.mean(results_dictionary[feature]['Pearson'])
        results[feature] = [R2, Pearson]
        print(f"{feature} - R2: {R2:.2f} Pearson: {Pearson:.2f}")
        
    return results


def log_transform(dataframe, cols_to_transform):
    """
    Perform log 10 transformation of specified columns
    Args:
        dataframe: full dataframe
        cols_to_transform: list of columns to perform transformation on
    """
    for col in cols_to_transform:
        if np.isin(col, dataframe.columns):
            dataframe.loc[:, col] = np.log10(dataframe[col])
    return dataframe


def log_transform_safely(dataFrame, cols_to_transform, min_value):
   """
       Performs log transform but sets any value that is negative infinty to a set min_value.
   """
   transformed = log_transform(dataFrame, cols_to_transform)
   negatives = transformed[cols_to_transform] == np.NINF
   try:
       for column in negatives.columns:
           transformed.loc[negatives[column], column] = min_value
   except:
       transformed.loc[negatives, cols_to_transform] = min_value
   return transformed