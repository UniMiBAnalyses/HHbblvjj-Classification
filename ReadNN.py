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
from sklearn.metrics import confusion_matrix
import itertools
import math
from sklearn.preprocessing import label_binarize


Path = "/home/wahid/Desktop/University/Python_Scripts"
Path_Model = Path + "/Model"
Path_Plot = Path + "/Plot"



def Significance(histo_S,histo_B):
    bins = histo_S.GetNbinsX()
    Significance = r.TH1F("sig","Significance",bins,0,1)
    c_sig = r.TCanvas("c2","c2",1000,1000,1000,1000)
    Significance.SetStats(0000)
    Significance.SetLineWidth(4)
    Significance.SetLineColor(r.kRed)
    for i in range(0,bins):
        int_S = histo_S.Integral(i,bins)
        int_B = histo_B.Integral(i,bins)
        if(int_B != 0):
            sig = int_S / math.sqrt(int_B)
            Significance.SetBinContent(i,sig)
    #Significance.SetXTitle("NN Response")
    #Significance.SetYTitle("Significance")
    #c_sig.SetTitle("Significance")
    Significance.Draw("APL")
    
    return c_sig



def plot_confusion_matrix(cm, names, title, cmap=plt.cm.Blues):
    #soglia per contrasto colore
    thresh = cm.max() / 1.5 
    #plotto per ogni cella della matrice il valore corrispondente
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{:0.4f}".format(cm[i, j]),
        horizontalalignment="center",
        color="white" if cm[i, j] > thresh else "black", fontsize=25)
    #plotto il resto
    #mp.rc('figure', figsize=(20,20), dpi=140)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=25)
    plt.colorbar()
    tick_marks = np.arange(len(names))
    plt.xticks(tick_marks, names, fontsize = 20)
    plt.yticks(tick_marks, names, fontsize = 20)
    plt.tight_layout()
    plt.ylabel('True label', fontsize = 20)
    plt.xlabel('Predicted label',fontsize = 20)
    plt.tight_layout()
    plt.draw()
    plt.savefig(Path_Plot+"/cm.pdf")
    print("Confusion_Matrix saved in {}".format(Path_Plot))
    
def Confusion_matrix(y, pred):
    fig = plt.figure(figsize=(10,10))
    cm = confusion_matrix(y, pred)
    np.set_printoptions(precision=2)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plot_confusion_matrix(cm_normalized,[0,1,2], title='Normalized confusion matrix')
    plt.draw()
    
    


#def Plotting_Roc_classes(n_classes,Y_test,pred):
    #i = 0
    #plt.figure(figsize=(20,10))
    #plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='b',label='0 pred power', alpha=.8)
    #while i < n_classes:
        #c = Y_test.argmax()
        ##c = (c==i).astype(int)
        #b = pred.argmax()
        ##b = (b==i).astype(int)
        #fp , tp, th = roc_curve(c, b)
        #roc = roc_auc_score(c, b)
        #plt.plot(fp, tp, label='ROC class %1.f (AUC = %0.3f)' %(i,roc))
        #plt.xlabel('False Positive Rate')
        #plt.ylabel('True Positive Rate')
        #plt.title('Receiver Operating Characteristic curve')
        #plt.legend(loc="lower right")
        #i+=1
        #plt.savefig(Path_Plot + "/ROC.pdf")
        #print("ROC-Curve saved in {}".format(Path_Plot))

        
    


X = np.load(Path+"/Data/X_data.npy")

Y = np.load(Path+"/Data/Y_data.npy")


X_train, x_test, Y_train, y_test = train_test_split(X,Y, test_size = 0.3)

print(X_train.shape,X.shape, Y_train.shape, Y.shape)
print(X_train,Y_train)


scaler = StandardScaler(copy = False, with_mean = True, with_std = True)
X_train = scaler.fit_transform(X_train)
x_test = scaler.transform(x_test)

json_file = open(Path_Model + "/Model.json","r")
loaded_model_json = json_file.read()
json_file.close()

loaded_model = model_from_json(loaded_model_json)
print("Model Loaded successfully!")

loaded_model.load_weights(Path_Model+"/Model.h5")

print("Loaded weights successfully!")


loaded_model.compile(loss = "binary_crossentropy",
                     optimizer = "adam",
                     metrics = ['acc'])

score = loaded_model.evaluate(x_test,y_test,verbose = 1)
print(score)

y_pred = loaded_model.predict(x_test)

print(y_pred)

#print(x_test.shape, y_test.shape)
#Y_pred = loaded_model.predict(Y_train)

histo_S = r.TH1F("h1","h1", 1000, 0, 1.01)

histo_B = r.TH1F("h2","h2", 1000, 0, 1.01)
k1 = r.TCanvas("c1","c1",50,50,1000,800)

for i in range(0,len(y_pred)):
    if(y_test[i] == 1):
        histo_S.Fill(y_pred[i])
    else:
        histo_B.Fill(y_pred[i])

histo_S.SetFillColor(r.kRed)
histo_S.SetLineColor(r.kRed)
histo_B.SetLineColor(r.kBlue)
histo_B.SetFillColor(r.kBlue)
histo_S.SetFillStyle(3002)
histo_B.SetFillStyle(3004)
histo_S.Draw("")
histo_B.Draw("same")
k1.SetLogy()
k1.SaveAs(Path_Plot + "/MLP_response.png","png")


#we have to approximate X_pred to 0 or 1 in order to computer de CM

y_pred = np.around(y_pred,0)

cm = Confusion_matrix(y_test, y_pred)

c_sig = Significance(histo_S,histo_B)
c_sig.Draw("")
c_sig.SaveAs(Path_Plot + "/Significance.png","png")


y = label_binarize(y_pred,classes=[0,1])



def Plotting_Roc_Curve(y_test,y):

    fp,tp,th = roc_curve(y_test,y)
    figure1 = plt.figure(figsize=(10,10))
    plt.plot(fp,tp)
    figure1.show()
    figure1.savefig(Path_Plot +"/ROC.pdf")
    print(fp,tp,th)

