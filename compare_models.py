import warnings
warnings.filterwarnings("ignore")

import sys
import os
import yaml
import copy
import torch
import pickle
import logging
import traceback
import numpy as np
import pandas as pd
import tensorflow as tf
import multiprocessing as mp
import matplotlib.pyplot as plt
import s3fs

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
sys.path.append("/glade/work/schreck/repos/GECKO_OPT/clean/gecko-ml")
# from geckoml.models import DenseNeuralNetwork, GRUNet
from geckoml.metrics import *
from geckoml.data import *
#from geckoml.box import *

from scipy.stats import pearsonr
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

import tqdm
import yaml





def get_stability(preds, stability_thresh):
    """
    Determine if any value has crossed the positive or negative magnitude of threshold and lable unstable if true
    Args:
        preds (pd.DataFrame): Predictions
        stability_thresh: Threshold to determine if an exp has gone unstable (uses positive and negative values)
    Returns:
        stable_exps (list)
        unstable_exps (list)
    """
    preds = preds.copy()
    preds['Precursor [ug/m3]'] = 10**(preds['Precursor [ug/m3]'])
    
    unstable = preds.groupby('id')['Precursor [ug/m3]'].apply(
        lambda x: x[(x > stability_thresh) | (x < -stability_thresh)].any())
    stable_exps = unstable[unstable == False].index
    unstable_exps = unstable[unstable == True].index

    return stable_exps, unstable_exps

def EarthMoverDist2D(Y1, Y2): 
    # https://stackoverflow.com/questions/57562613/python-earth-mover-distance-of-2d-arrays
    d = cdist(Y1, Y2)
    assignment = linear_sum_assignment(d)
    return d[assignment].sum() / Y1.shape[0]


def metrics(truth, preds, stable_thresh = 1.0):
        
    for i, l in zip([1,2,3], ["prec", "gas", "aero"]):
        t = truth.iloc[:, i]
        p = preds.iloc[:, i]
        
        if i == 1:
            t = 10**t
            p = 10**p
        
        mae = mean_absolute_error(t,p)
        r2 = r2_corr(t, p)
        hd = hellinger_distance(t, p)
        pearson = pearsonr(t,p)[0]

        experiments = int(t.shape[0] / 1439)
        t = t.to_numpy().reshape((experiments, 1439))
        p = p.to_numpy().reshape((experiments, 1439))
        emd = EarthMoverDist2D(t, p)
        
        print(f"{l} -- ", f"MAE: {mae:.5f} R2: {r2:.3f} Pearson {pearson:.3f} Hellinger: {hd:.5f} EMD: {emd:.3f}") 

def plot(truth, preds, fontsize = 14, prec_lim = 0.085, gas_lim = 0.034, aero_lim = 0.0105):

    plt.figure(figsize=(12,8))

    for k, exp in enumerate(["Exp1709", "Exp1632", "Exp1769"]):
        colors = ["r", "g", "b"]

        #exp = "Exp1769"
        c1 = (preds["id"] == exp)
        c2 = (truth["id"] == exp)

        #plt.title(f"{exp}")

        plt.subplot(3, 3, k + 1)
        plt.plot(preds[c1]["Time [s]"] / 3600, 
                 10**preds[c1]["Precursor [ug/m3]"], 
                 c = colors[k], 
                 linewidth = 3)
        plt.plot(truth[c2]["Time [s]"] / 3600, 
                 10**truth[c2]["Precursor [ug/m3]"], 
                 ls = '--', 
                 c = 'k', 
                 linewidth = 3)
        if k == 0:
            plt.ylabel("Precursor", fontsize=fontsize)

        plt.legend(["Pred", "True"], fontsize=fontsize)
        plt.ylim([0, prec_lim])
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.title(exp, fontsize=fontsize)
        
    #     plt.tick_params(axis = "both", left='on', top='off', right='off', bottom='on', 
    #                     labelleft='off' if k > 0 else 'on', 
    #                     labeltop='off', 
    #                     labelright='off', 
    #                     labelbottom='off' if k != 2 else 'on',
    #                     direction = "in")

        plt.subplot(3, 3, k + 4)
        plt.plot(preds[c1]["Time [s]"] / 3600, preds[c1]["Gas [ug/m3]"], c = colors[k], linewidth = 3)
        plt.plot(truth[c2]["Time [s]"] / 3600, truth[c2]["Gas [ug/m3]"], ls = '--', c = 'k', linewidth = 3)
        if k == 0:
            plt.ylabel("Gas", fontsize=fontsize)
        #plt.legend(["Pred", "True"])
        #plt.xlim([0, 1440])
        plt.ylim([0, gas_lim])
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)

        plt.subplot(3, 3, k + 7)
        plt.plot(preds[c1]["Time [s]"] / 3600, preds[c1]["Aerosol [ug_m3]"], c = colors[k], linewidth = 3)
        plt.plot(truth[c2]["Time [s]"] / 3600, truth[c2]["Aerosol [ug_m3]"], ls = '--', c = 'k', linewidth = 3)
        if k == 0:
            plt.ylabel("Aerosol", fontsize=fontsize)
        plt.xlabel("Time (hr)", fontsize=fontsize)
        #plt.legend(["Pred", "True"])
        plt.ylim([0, aero_lim])
        #plt.xlim([0, 1440])
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)

    plt.tight_layout()
    plt.show()
    
def box_val(mod, exps, num_timesteps, in_array, env_array, y_scaler, output_cols, out_val, stable_thresh = 1.0):
    
    # use initial condition @ t = 0 and get the first prediction
    pred_array = np.empty((len(exps), 1439, 3))
    pred = mod.predict(in_array[:, 0, :])
    pred_array[:, 0, :] = pred

    # use the first prediction to get the next, and so on for num_timesteps
    for i in tqdm.tqdm(range(1, num_timesteps)):
        temperature = in_array[:, i, 3:4]
        static_env = env_array[:, -5:]
        new_input = np.block([pred, temperature, static_env])
        #pred = mod(new_input, training=False)
        pred = mod.predict(new_input)
        pred_array[:, i, :] = pred

    # loop over the batch to fill up results dict
    results_dict = {}
    for k, exp in enumerate(exps):
        results_dict[exp] = pd.DataFrame(y_scaler.inverse_transform(pred_array[k]), columns=output_cols[1:-1])
        results_dict[exp]['id'] = exp
        results_dict[exp]['Time [s]'] = out_val['Time [s]'].unique()
        results_dict[exp] = results_dict[exp].reindex(output_cols, axis=1)

    preds = pd.concat(results_dict.values())
    truth = out_val.loc[out_val['id'].isin(exps)]
    truth = truth.sort_values(['id', 'Time [s]']).reset_index(drop=True)
    preds = preds.sort_values(['id', 'Time [s]']).reset_index(drop=True)
    
    stable_exps, failed_exps = get_stability(preds, stable_thresh)
        
    c1 = ~truth["id"].isin(failed_exps)
    c2 = ~preds["id"].isin(failed_exps)
    box_mae = mean_absolute_error(preds[c2].iloc[:, 2:-1], truth[c1].iloc[:, 2:-1])
    
    return box_mae, truth, preds, failed_exps


# if __name__ == "__main__":
#     main()