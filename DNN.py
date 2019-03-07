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
import keras
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
from keras.optimizers import Adam, SGD
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
from keras import backend


parser = argparse.ArgumentParser()
parser.add_argument('-t', '--tree', type=str, required=True, help="type positive or negative for delta") 
args = parser.parse_args()

def rmse(y_true, y_pred):
	return np.sqrt(metrics.mean_squared_error(pred,y_test))

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
x,y = to_xy(data, data.shape[1]-1)
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
print("Dataset shape:  {}" .format(data.shape[1]-1))

#>>>>>>>>>>>>>>>>>Y mass <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
h = r.TH1F("h","h", 60,50,150)
i = 0
while i < len(y):
    h.Fill(y[i])
    i += 1
c = r.TCanvas("c", "c", 50,50,1000,800)
h.Draw("hist")
c.Draw()
c.SaveAs(output_dir + "/Y.png")
hs = r.TH1F("hs","hs", 60,50,150)
i = 0
while i < len(y_test):
    hs.Fill(y_test[i])
    i += 1
cs = r.TCanvas("cs", "cs", 50,50,1000,800)
hs.Draw("hist")
cs.Draw()
cs.SaveAs(output_dir + "/Y_test.png")


def Modeloptimizer(num_dense_layers, num_dense_nodes, loss_func, first_layer):
    print(">>> Creating model...")
    print(">>> Dense Layers: {}".format(num_dense_layers))
    print(">>> Dense Neurons: {}".format(num_dense_nodes))
    #print(">>> Dropout Rate: {}".format(dropout_rate))
    print(">>> Loss: {}".format(loss_func))
    print(">>> First layer: {}".format(first_layer))
    #print(">>> Learning rate: {}".format(learning_rate))
    model = Sequential()
    #print(">>> input dim {}".format(data.shape[1]-1))
    model.add(Dense(units = first_layer ,input_dim=data.shape[1]-1, activation="relu"))
    #model.add(Dropout(dropout_rate))
    for i in range(num_dense_layers):
        # Name of the layer. This is not really necessary
        # because Keras should give them unique names.
        name = 'layer_dense_{0}'.format(i+1)

        # Add the dense / fully-connected layer to the model.
        # This has two hyper-parameters we want to optimize:
        # The number of nodes and the dropout.
        model.add(Dense(units=num_dense_nodes,
                        activation="relu",
                        name=name))
        #model.add(Dropout(dropout_rate))
        
    #last layer for regression must be one, no activation gives pure scalar as output.
    model.add(Dense(1))
    #opt = Adam(lr = learning_rate)
    model.compile(optimizer='adam',
                  loss=loss_func,
                  metrics=['mse'])


    return model
"""

def Modeloptimizer(loss_func, first_layer):
    print(">>> Creating model...")
    print(">>> First Layer: {}".format(first_layer))
    print(">>> Loss: {}".format(loss_func))
    model = Sequential()
    #print(">>> input dim {}".format(data.shape[1]-1))
    model.add(Dense(units = first_layer ,input_dim=data.shape[1]-1, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(units = 15 , activation = "relu"))
    model.add(Dropout(0.3))
    model.add(Dense(units = 15 , activation = "relu"))
    model.add(Dropout(0.3))
    model.add(Dense(units = 15 , activation = "relu"))
    model.add(Dropout(0.3))
    model.add(Dense(units = 15 , activation = "relu"))
    model.add(Dropout(0.3))
        
    #last layer for regression must be one, no activation gives pure scalar as output.
    model.add(Dense(1))
    
    model.compile(optimizer='adam',
                  loss=loss_func,
                  metrics=['mse'])


    return model
"""

#>>>>>>>defining Hyper parameter research space<<<<<<<<<<<<

print(">>>Defining grid search param")

num_dense_layers_candidates = [2]
num_dense_nodes_candidates = [18]  
dropout_rate_candidates = [0.1]
loss_candidates = ["mean_absolute_error"]
first_candidates = [14]
lear_rate = [0.2]



parameters = {'num_dense_layers': num_dense_layers_candidates,
              'num_dense_nodes': num_dense_nodes_candidates,
              #'dropout_rate': dropout_rate_candidates,
              'loss_func': loss_candidates,
              'first_layer': first_candidates,
              #'learning_rate': lear_rate
              }

"""
loss_candidates = ["mean_absolute_error", "mean_squared_error", "mean_absolute_percentage_error", "logcosh", "mean_squared_logarithmic_error"]
first_layer_candidates = [20,25]
"""

early_stop = EarlyStopping(monitor='mean_squared_error', min_delta=1e-15, patience=15, verbose=1, mode='auto', baseline=None)
callbacks=[early_stop]
regressor = KerasRegressor(build_fn=Modeloptimizer, epochs=1000, verbose=0)
scorer = make_scorer(metrics.mean_absolute_error, greater_is_better = False)


print(">>>Grid Search begins")
validator = GridSearchCV(regressor,
                         param_grid= parameters,
                         fit_params={'callbacks': [early_stop]},
                         scoring=scorer,
                         refit='my_score',
                         return_train_score=True,
                         n_jobs=-1)

"""
validator = GridSearchCV(regressor,
                         param_grid={'first_layer': first_layer_candidates,
                                     'loss_func': loss_candidates},
                                     
                        fit_params={'callbacks': [early_stop]},
                        scoring=scorer,
                        refit='my_score',
                        return_train_score=True,
                        n_jobs=1)
"""
print(">>> x train shape: {}".format(x_train.shape))

grid_result = validator.fit(x_train, y_train)
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
print('Best score %0.3f' % grid_result.best_score_)
print("All results:")
print(grid_result.grid_scores_)
print('Best parameters:')
print(grid_result.best_params_)


best_model = validator.best_estimator_.model

"""
metric_values = best_model.evaluate(x_test, y_test)
metric_names = best_model.metrics_names
print(metric_names)
print(metric_values)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
print(grid_result.grid_scores_)
"""
#>>>>>>>>>Saving graphs and grid result<<<<<<<<<<<<<<<

pred = best_model.predict(x_test)
pred = np.concatenate(pred)
score = np.sqrt(metrics.mean_squared_error(pred,y_test))
print("Best DNN Score (RMSE): {}".format(score))

f = open(output_dir + "/DNN_grid.txt", "w")
f.write("{}\n".format("Best score:"))
f.write("MAE negated: {}\n".format(grid_result.best_score_))
f.write("Best DNN Score (RMSE): {}\n".format(score))
f.write("{}\n".format(grid_result.best_params_))
f.write("{}\n".format("All scores:"))
f.write("{}\n".format(grid_result.grid_scores_))

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
legend.Draw("same")
r.gStyle.SetOptStat(0)
c3.Draw()
c3.SaveAs(output_dir + "/overlap_dev.png")

#>>>>>>>>>>>>Scatter Plot<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
i = 0
if args.tree == "positive":
    h3 = r.TH2F("Scatter Plot Positive", "Scatter Plot Positive", 60, 50, 150, 60, 50, 150)
else:
    h3 = r.TH2F("Scatter Plot Negative", "Scatter Plot Negative", 60, 50, 150, 60, 50, 150)
while i < len(pred):
    h3.Fill(y_test[i], pred[i])
    i += 1
c4 = r.TCanvas("c2", "c2", 50,50,1000,800)
h3.Draw("COLZ")
h3.GetXaxis().SetTitle("m_{l#nu,MC}")
h3.GetYaxis().SetTitle("m_{l#nu,DNN}")
c4.Draw()
c4.SaveAs(output_dir + "/ScatterPlot.png")

#>>>>>>>>>>>>Scatter Plot2<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

pred1 = best_model.predict(x)
pred1 = np.concatenate(pred1)
if args.tree == "positive":
    h4 = r.TH2F("Scatter Plot Positive2", "Scatter Plot Positive2", 60, 50, 150, 60, 50, 150)
else:
    h4 = r.TH2F("Scatter Plot Negative2", "Scatter Plot Negative2", 60, 50, 150, 60, 50, 150)
while i < len(pred1):
    h4.Fill(y[i], pred1[i])
    i += 1
c5 = r.TCanvas("c5", "c5", 50,50,1000,800)
h4.Draw("COLZ")
h4.GetXaxis().SetTitle("m_{l#nu,MC}")
h4.GetYaxis().SetTitle("m_{l#nu,DNN}")
c5.Draw()
c5.SaveAs(output_dir + "/ScatterPlot2.png")


