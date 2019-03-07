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
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA


def to_xy(df,target):
    y = df[:,target]
    x = np.delete(df, target, 1)
    return x,y


Path = "/home/wahid/Desktop/University/Python_Scripts"
Path_Model = Path + "/Model"
Path_Plot = Path + "/Plot"

print("Loading the datasets")
Sig = np.load(Path+"/Data/Signal_pca.npy")
Bkg = np.load(Path+"/Data/Background_pca.npy")
print("Loaded!")



wanted_variables =np.array( ["mww","deltar_ljj","deltaphi_ljj","bb_pt","deltaphi_bbljj","deltar_bbljj","ww_pt","Ptm","deltar_bb","deltaphi_bb","Ht"])


Signal = np.array([])
Background = np.array([])


string_var = Sig[0,:]
indices = []

for i in wanted_variables:
    for j in range(0,len(string_var)):
                       if(i == string_var[j]):
                           indices.append(j)


signal = np.ones(len(Sig[1:,1]))
bkg = np.zeros(len(Bkg[1:,0]))
#data_S[:,1]
for i in indices:
    signal = np.c_[signal,Sig[1:,i]]
    bkg = np.c_[bkg,Bkg[1:,i]]


Signal = signal[:,1:]
Background = bkg[:,1:]


for var in wanted_variables:
    for i in range(0,len(Sig[0,:])):
        if(var == Sig[0,i]):
            Signal_tot = np.vstack(Sig[1:,i])  
            

for var in wanted_variables:
    for i in range(0,len(Bkg[0,:])):
        if(var == Bkg[0,i]):
            Background_tot = np.vstack(Bkg[1:,i])     
            
            
scaler = StandardScaler(copy = False, with_mean = True, with_std = True)
Signal = scaler.fit_transform(Signal)
Background = scaler.transform(Background)

            
#sig0 = Signal[:,0].astype(np.float)
            
#sig1 = Signal[:,1].astype(np.float)
            
#sig2 = Signal[:,2].astype(np.float)


#sig0 = array("f",sig0)

#sig1 = array("f",sig1)

#sig2 = array("f",sig2)



#bkg0 = Background[:,0].astype(np.float)
            
#bkg1 = Background[:,1].astype(np.float)
            
#bkg2 = Background[:,2].astype(np.float)


#bkg0 = array("f",bkg0)

#bkg1 = array("f",bkg1)

#bkg2 = array("f",bkg2  )
##bkg = Background.astype(np.float)


##print(sig)



#h2_s = r.TH3F("h2","h2",100,0,5,100,-3,3,100,-3,3)

#h2_b = r.TH3F("h2","h2",100,0,5,100,-3,3,100,-3,3)


#for i in range(0,len(sig0)):
    #h2_s.Fill(sig0[i],sig1[i],sig2[i])


#for i in range(0,len(bkg0)):
    #h2_b.Fill(bkg0[i],bkg1[i],bkg2[i])


    
##app = r.TApplication("app",0,0)
##print(h2.GetEntries())
#c1 = r.TCanvas("c1","c1",50,50,1000,800)
#h2_s.SetMarkerColor(r.kRed)
#h2_b.SetMarkerColor(r.kBlue)
#h2_s.Draw("")
#h2_b.Draw("same")
#c1.Draw()
#c1.Print()
#print ("Scrivi s per chiudere il canvas: " )    #serve a mantenere aperto il canvas
#ss1 = input()    
#if (ss1=="s"):  
    #print ("canvas chiuso")


SIG = np.load(Path+"/Data/Signal.npy")
BKG = np.load(Path+"/Data/Background.npy")

dataset = np.vstack((SIG,BKG))

X,Y = to_xy(dataset,dataset.shape[1]-1)

scaler = StandardScaler(copy = False, with_mean = True, with_std = True)
X = scaler.fit_transform(X)
#Y = scaler.transform(Y)




print(X,Y)

pca = PCA(n_components=5)

principalComponents = pca.fit_transform(X)

principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2','principal component 3','principal component 4',"principal component 5"])

ydf = pd.DataFrame(data = Y
             , columns = ['target'])

finalDf = pd.concat([principalDf, ydf['target']], axis = 1)


pc1 = finalDf["principal component 1"].values.tolist()
pc2 = finalDf["principal component 2"].values.tolist()
pc3 = finalDf["principal component 3"].values.tolist()
pc4 = finalDf["principal component 4"].values.tolist()
pc5 = finalDf["principal component 5"].values.tolist()


pc1 = array("f",pc1)
pc2 = array("f",pc2)
pc3 = array("f",pc3)
pc4 = array("f",pc4)
pc5 = array("f",pc5)

h_pca_S = r.TH3F("pca","pca",100,-10,10,100,-10,10,100,-10,10)
h_pca_B = r.TH3F("pca","pca",100,-10,10,100,-10,10,100,-10,10)

for i in range(0,len(SIG)):
    h_pca_S.Fill(pc1[i],pc2[i],pc3[i])

for i in range(0,len(BKG)):
    h_pca_B.Fill(pc1[len(SIG)+i],pc2[len(SIG)+i],pc3[len(SIG)+i]) #Sono ordinati quindi il fondo parte dopo il segnale

c1 = r.TCanvas("c1","c1",50,50,1000,800)
h_pca_S.SetMarkerColor(r.kRed)
h_pca_B.SetMarkerColor(r.kBlue)
h_pca_S.Draw("")
h_pca_B.Draw("same")
c1.Draw()
#c1.Print()
#print ("Scrivi s per chiudere il canvas: " )    #serve a mantenere aperto il canvas
#ss1 = input()    
#if (ss1=="s"):  
    #print ("canvas chiuso")


h_pca1_s = r.TH1F("h_pca5","h_pca5",100,-10,10)

h_pca1_b = r.TH1F("h_pca5","h_pca5",100,-10,10)

for i in range(0,len(SIG)):
    h_pca1_s.Fill(pc5[i])
    

for i in range(0,len(BKG)):
    h_pca1_b.Fill(pc5[len(SIG)+i])
    
h_pca1_s.Scale(1/h_pca1_s.Integral())
h_pca1_b.Scale(1/h_pca1_b.Integral())

c_pca = r.TCanvas("c_pca","c_pca",50,50,1000,800)
h_pca1_s.SetFillStyle(3002)
h_pca1_b.SetFillStyle(3004)
h_pca1_s.SetFillColor(r.kRed)
h_pca1_b.SetFillColor(r.kBlue)
h_pca1_s.SetLineColor(r.kRed)
h_pca1_b.SetLineColor(r.kBlue)
h_pca1_s.Draw("histo")
h_pca1_b.Draw("histo SAME")
c_pca.Draw("")
c_pca.Print()
print ("Scrivi s per chiudere il canvas: " )    #serve a mantenere aperto il canvas
ss1 = input()    
if (ss1=="s"):  
    print ("canvas chiuso")

