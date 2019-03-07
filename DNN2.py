"""
>>>INTRO<<<

This program was built to search in the hyper-parameters space for the best Deep Neural Network configuration with a Grid Search with 3-fold cross validation in order to solve neutrino momentum reconstruction. Events are taken from a .root TTree containing 2 branches, one for solutions with positive discriminant, the second with negative discriminant. Thanks to argparse we can select with which branch we want to work. Analytical method is then compared with the DNN result (in deviations with respect to the Monte Carlo information). Hyper parameter space is fully customizable but pay close attention to dimensionality curse. It is suggested to tune few hyper parameters (such as number of dense layers and neurons) and then, from the best model, tune other hyper parameters such as learning rate, dropout rate, loss function and so on. The best model obtained can be saved and loaded. Here we plot directly all the deviation distributions and comparision.

To run the program:

python DNN.py -t positive 

or

python DNN.py -t negative


Giacomo Boldrini
"""


import ROOT as r
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib as mp
import itertools
import sklearn
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers import Dropout
import sys
import math
from tensorflow.python.keras import backend as K
from array import array
import argparse
from sklearn import svm, datasets
from sklearn.model_selection import KFold
from scipy import interp
from keras.callbacks import History 
from scipy.stats import pearsonr
from tqdm import tqdm
import pandas as pd
from keras.callbacks import *
from copy import deepcopy
import random
from keras.losses import *
from keras.optimizers import Adam
from sklearn import metrics
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from scipy.stats import zscore
import random
random.seed(10)
import argparse
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.metrics import make_scorer

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--tree', type=str, required=True, help="type positive or negative for delta") 
args = parser.parse_args()


def to_xy(df, target):
    y = df[:,target]
    x = np.delete(df, target, 1)
    return x,y

def mkdir_p(mypath):
    #crea una directory

    from errno import EEXIST
    from os import makedirs,path

    try:
        makedirs(mypath)
    except OSError as exc: 
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else: raise
    
#>>>>>>>>>>Reading and creating dataset<<<<<<<<<<<<<<<<<

if args.tree == "positive":
    tree = "tree_pos;1"
elif args.tree == "negative":
    tree = "tree_neg;1"
else:
    print(">>>No tree found from args input")

print(">>>Reading tree {}".format(tree))
f = r.TFile("/home/wahid/Desktop/University/Python_Scripts/Data/Anna.root")
output_dir = "/home/wahid/Desktop/University/Python_Scripts/Data/DNN_results"
mkdir_p(output_dir)

print(">>>Creating dataset")
myTree = f.Get(tree)
l = []
Mcal = []
Mmc = []
for event in myTree:
    l.append([event.l_px, event.l_py, event.l_pz, event.l_E, event.n_px, event.n_py, event.n_pz, event.n_E, event.q1_px, event.q1_py, event.q1_pz, event.q1_E, event.q2_px, event.q2_py, event.q2_pz, event.q2_E, event.q3_px, event.q3_py, event.q3_pz, event.q3_E, event.q4_px, event.q4_py, event.q4_pz, event.q4_E, event.M])
    Mcal.append(event.M_calc)
    Mmc.append(event.M)

data = np.array(l)
print("Dataset shape created")
print(data.shape)
random.shuffle(data)
x,y = to_xy(data, data.shape[1]-1)
#x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.5,random_state=42)
i = round((x.shape[0]-1)*0.5)
x_train = x[:i]
y_train = y[:i]
x_test = x[i:]
y_test = y[i:]
Mcal_test = Mcal[i:]
Mmc_test = Mmc[i:]
print(">>>x shape: {}".format(x.shape))
print(">>>y shape: {}".format(y.shape))
print(">>>x_train shape: {}".format(x_train.shape))
print(">>>y_train shape: {}".format(y_train.shape))
print(">>>x_test shape: {}".format(x_test.shape))
print(">>>y_test shape: {}".format(y_test.shape))
print("<<<< datashape: {}" .format(data.shape[1]-1))


  

#>>>>>>>defining Hyper parameter research space<<<<<<<<<<<<

print(">>>Defining grid search param")
"""
num_dense_layers_candidates = [3,4,5]
num_dense_nodes_candidates = [10,12,15,20,30]  
dropout_rate_candidates = [0.2,0.3,0.4]
#loss_candidates = ["mean_absolute_error", "mean_squared_error"]
"""
#first_layer_candidates = [10, 15, 20, 25, 30, 35]

#loss_candidates = ["mean_absolute_error", "mean_squared_error", "mean_absolute_percentage_error", "logcosh", "mean_squared_logarithmic_error"]

#scorer = {'my_score': make_scorer(mean_squared_error)}

early_stop = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=10, verbose=1, mode='auto')

#regressor = KerasRegressor(build_fn=Modeloptimizer, epochs=1000, verbose=0)

def grid_search_wrapper(refit_score='my_score'):
    """
    fits a GridSearchCV classifier using mean squared error for optimization
    prints classifier performance metrics
    """
    
    validator = GridSearchCV(regressor,
                         param_grid={'first_layer': first_layer_candidates,
                                     'loss_func': loss_candidates},
                                     
                        fit_params={'callbacks': [early_stop]},
                        scoring=scorer,
                        refit=refit_score,
                        return_train_score=True,
                        n_jobs=-1)
                         
    
    #grid_result = validator.fit(x_train, y_train)
    #best_model = validator.best_estimator_.model
    #pred = best_model.predict(x_test)
    #pred = np.concatenate(pred)
    #print('Best params for {}'.format(refit_score))
    #print(grid_result.best_params_)

    return validator


print(">>>Grid Search begins")

validator = grid_search_wrapper

print(type.validator)
#validator.fit(X_train, Y_train)
#best_model = validator.best_estimator_.model


#>>>>>>>>>Saving graphs and grid result<<<<<<<<<<<<<<<

pred = best_model.predict(x_test)
pred = np.concatenate(pred)
score = np.sqrt(metrics.mean_squared_error(pred,y_test))
print("Best DNN Score (RMSE): {}".format(score))

f = open(output_dir + "/DNN_grid.txt", "w")
f.write("{}\n".format(grid_result.best_score_))
f.write("{}\n".format(grid_result.best_params_))
f.write("Best DNN Score (RMSE): {}\n".format(score))
f.close()

#>>>>>>><Computing deviation monte carlo and calculated on the test set<<<<<<<<<<

deviation_calc = []
i=0
while i < len(Mcal_test):
    deviation_calc.append((Mcal_test[i]-Mmc_test[i])/Mmc_test[i])
    i+=1
hm = r.TH1F("calculated vs monte carlo", "calculated vs monte carlo", 150, -0.5,0.5)
hm.SetLineColor(r.kBlue)
hm.SetLineWidth(2)
hm.SetFillColor(r.kBlue)
hm.SetFillStyle(3003)
for i in deviation_calc:
    hm.Fill(i)
c1 = r.TCanvas("canvas","canvas", 50,50, 1000,800)
hm.Draw("hist")
c1.Draw()
c1.SaveAs(output_dir + "/calc_deviation.png")

#>>>>>>>>Graphs for DNN deviation from monte carlo results<<<<<<<<<<<<<<<<<<<<<<

i = 0
histo = r.TH1F("DNN deviation", "DNN deviation", 150, -0.5, 0.5)
histo.SetLineColor(r.kRed)
histo.SetLineWidth(2)
histo.SetFillColor(r.kRed)
histo.SetFillStyle(3003)
while i < len(pred):
    dev = (pred[i]-y_test[i])/y_test[i]
    histo.Fill(dev)
    i += 1
c2 = r.TCanvas("c1", "c1", 50, 50, 1000, 800)
histo.Draw("hist")
c2.Draw()
c2.SaveAs(output_dir + "/model_deviation.png")

#>>>>>>>>>>>>Overlapping the two graphs<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
legend = r.TLegend(.9,.9,0.7,0.7)
legend.AddEntry(histo)
legend.AddEntry(hm)
c3 = r.TCanvas("c2", "c2", 50, 50, 1000, 800)
hm.Draw("hist")
histo.Draw("hist same")
r.gStyle.SetOptStat(0)
#legend.Draw()
c3.Draw()
c3.SaveAs(output_dir + "/overlap_dev.png")
