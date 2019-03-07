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
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, auc
import pickle
from keras.models import model_from_json




Path = "/home/wahid/Desktop/University/Python_Scripts/"

def mkdir_p(mypath):
    from errno import EEXIST
    from os import makedirs,path
    try:
        makedirs(mypath)
    except OSError as exc: 
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else: raise

def to_xy(df,target):
    y = df[:,target]
    x = np.delete(df, target, 1)
    return x,y

def acc_plotter(history):
    plt.figure(figsize=(10,5))
    plt.plot(history.history['acc'], label = "train_accuracy")
    plt.plot(history.history['val_acc'], label="val_accuracy")
    plt.legend(['acc_train', 'acc_test'], loc='lower right')
    plt.xlabel('epochs')
    plt.ylabel('acc')
    plt.show()

def loss_plotter(history):
    plt.figure(figsize=(10,5))
    plt.plot(history.history['loss'], label = "train_loss")
    plt.plot(history.history['val_loss'], label="val_loss")
    plt.legend(['loss_train', 'loss_test'], loc='upper right')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.show()
    
    
print("Loading the datasets")
Sig = np.load(Path+"/Data/Signal.npy")
Bkg = np.load(Path+"/Data/Background.npy")
print("Loaded!")

#Sig_shuffled = sorted(Sig, key=lambda k: random.random())
#Bkg_shuffled = sorted(Bkg, key=lambda k: random.random())

dataset = np.vstack((Sig,Bkg))
print(dataset.shape)

random.shuffle(dataset)

X,Y = to_xy(dataset,dataset.shape[1]-1)


np.save("/home/wahid/Desktop/University/Python_Scripts/Data/X_data.npy",X)
np.save("/home/wahid/Desktop/University/Python_Scripts/Data/Y_data.npy",Y)





#Starting with the neural network
X_train, x_test, Y_train, y_test = train_test_split(X,Y, test_size = 0.3)
#X_train, X_validation, Y_train, Y_validation = train_test_split(X,Y, test_size = 0.3)

print(X_train.shape,X.shape, Y_train.shape, Y.shape)
print(X_train,Y_train)


scaler = StandardScaler(copy = False, with_mean = True, with_std = True)
X_train = scaler.fit_transform(X_train)
x_test = scaler.transform(x_test)




def create_model():
	# create model
	model = Sequential()
	model.add(Dense(5, input_dim=dataset.shape[1]-1, activation='tanh'))
	
	
	model.add(Dense(1, activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

'''
model = KerasClassifier(build_fn=create_model, verbose=1)
# define the grid search parameters
batch_size = [5,10]
epochs = [5,10,20]
param_grid = dict(batch_size=batch_size, epochs=epochs)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
grid_result = grid.fit(X_train, Y_train)


print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
'''






model = Sequential()

model.add(Dense(units = 5, input_dim=dataset.shape[1]-1, activation="tanh"))
model.add(Dense(5))
model.add(Dense(5))

#Last layer must be one with sigmoid activation function

model.add(Dense(1, activation = "sigmoid"))

early_stop = EarlyStopping(monitor = "val_loss" ,min_delta=1e-9, patience=5, verbose=1, mode='auto')

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['acc'])


history = model.fit(X_train,Y_train, epochs=100, verbose = 1, validation_data=(x_test,y_test), callbacks = [early_stop])

filename = Path + 'Model/'

scores = model.evaluate(x_test,y_test, verbose = 1)

print(model.metrics_names,scores)


#serilize model to JSON

model_json = model.to_json()
with open(filename+"model.json","w") as json_file:
              json_file.write(model_json)

#serialize weights to HDF5

model.save_weights(filename + "model.h5")

#mkdir_p(Path+"/Plot")

acc_plotter(history)

loss_plotter(history)





pred = model.predict(x_test)




'''
def Modeloptimizer(loss_func, first_layer):
    print(">>> Creating model...")
    print(">>> First Layer: {}".format(first_layer))
    print(">>> Loss: {}".format(loss_func))
    model = Sequential()
    #print(">>> input dim {}".format(data.shape[1]-1))
    model.add(Dense(units = first_layer ,input_dim=dataset.shape[1]-1, activation="tanh"))
    #model.add(Dropout(0.3))
    model.add(Dense(units = 5 , activation = "tanh"))
    #model.add(Dropout(0.3))
    model.add(Dense(units = 5 , activation = "tanh"))
    #model.add(Dropout(0.3))
    #model.add(Dense(units = 15 , activation = "relu"))
    #model.add(Dropout(0.3))
    #model.add(Dense(units = 15 , activation = "relu"))
    #model.add(Dropout(0.3))
        
    ##last layer for regression must be one, no activation gives pure scalar as output.
    model.add(Dense(1))
    
    model.compile(optimizer='adam',
                  loss=loss_func,
                  metrics=['acc'])


    return model

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
                         
    
    grid_result = validator.fit(X_train, Y_train)
    best_model = validator.best_estimator_.model
    pred = best_model.predict(x_test)
    pred = np.concatenate(pred)
    print('Best params for {}'.format(refit_score))
    print(grid_result.best_params_)

    return validator
'''



'''
first_layer_candidates = [3,5,7,10,12,15]

loss_candidates = ["binary_crossentropy","logcosh"]

scorer = {'my_score': make_scorer(binary_crossentropy)}

early_stop = EarlyStopping(monitor='mean_squared_error', min_delta=1e-7, patience=10, verbose=1, mode='auto', baseline=None)
callbacks=[early_stop]
regressor = KerasRegressor(build_fn=Modeloptimizer, epochs=200, verbo
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
                         
    
    grid_result = validator.fit(X_train, Y_train)
    best_model = validator.best_estimator_.model
    pred = best_model.predict(x_test)
    pred = np.concatenate(pred)
    print('Best params for {}'.format(refit_score))
    print(grid_result.best_params_)

se=1)



print(">>>Grid Search begins")

validator = grid_search_wrapper()
validator.fit(X_train, Y_train)
best_model = validator.best_estimator_.model

pred = best_model.predict(x_test)
pred = np.concatenate(pred)
score = np.sqrt(metrics.mean_squared_error(pred,y_test))
print("Best DNN Score (RMSE): {}".format(score))
'''



#The last layer must have only one neuron!

#model.compile(optimizer='adam',
              #loss='binary_crossentropy',
              #metrics=['accuracy'])


#model.fit(X_train,Y_train, epochs=30, verbose = 1)
