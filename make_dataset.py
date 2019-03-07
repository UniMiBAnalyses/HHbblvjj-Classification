import ROOT as r
import keras 
import numpy as np
import matplotlib.pyplot as plt
import math as mth 
from tqdm import tqdm


file_root_sig = r.TFile("./Data/HH_bis.root","read")
file_root_bkg = r.TFile("./Data/ttbar-semi-lep.root","read")

Branch_Names = ["lep_pt","lep_eta","lep_E","lep_Et","n_pt",
              "j1_pt","j1_eta","j2_eta","j2_pt","b1_pt","b1_eta",
               "b2_pt","mjj","j_pt","j_Et","mww","ww_pt","ww_Et","mbb",
               "bb_pt","bb_Et","Ht","Htnu","Ptm","deltaphi_bb","deltaphi_ljj",
               "deltaphi_bbljj","deltaeta_bb","deltaeta_ljj","deltaeta_bbljj",
               "deltar_bb","deltar_ljj","deltar_bbljj"]



print(">>>Creating data")
myTree_S = file_root_sig.Get("tree")
myTree_B = file_root_bkg.Get("tree")

l_S = []
l_B = []

print("Loading signal events")

with tqdm(total=file_root_sig.Get("tree").GetEntries()) as pbar:
    for event in myTree_S:
        l_S.append([event.lep_pt, event.lep_eta, event.lep_E, event.lep_Et, event.n_pt, event.j1_pt, event.j1_eta, event.j2_eta, event.j2_pt, event.b1_pt, event.b1_eta, event.b2_pt, event.mjj, event.j_pt, event.j_Et, event.mww, event.ww_pt, event.ww_Et, event.mbb, event.bb_pt, event.bb_Et, event.Ht, event.Htnu, event.Ptm, event.deltaphi_bb, event.deltaphi_ljj,event.deltaeta_bbljj,event.deltaeta_bb,event.deltaeta_ljj,event.deltaeta_bbljj,event.deltar_bb,event.deltar_ljj,event.deltar_bbljj])
        pbar.update(1)

print("Loading complete!")
        
print("\nLoading background events")
with tqdm(total=file_root_bkg.Get("tree").GetEntries()) as pbar:
    for event in myTree_B:
        l_B.append([event.lep_pt, event.lep_eta, event.lep_E, event.lep_Et, event.n_pt, event.j1_pt, event.j1_eta, event.j2_eta, event.j2_pt, event.b1_pt, event.b1_eta, event.b2_pt, event.mjj, event.j_pt, event.j_Et, event.mww, event.ww_pt, event.ww_Et, event.mbb, event.bb_pt, event.bb_Et, event.Ht, event.Htnu, event.Ptm, event.deltaphi_bb, event.deltaphi_ljj,event.deltaeta_bbljj,event.deltaeta_bb,event.deltaeta_ljj,event.deltaeta_bbljj,event.deltar_bb,event.deltar_ljj,event.deltar_bbljj])
        pbar.update(1)
print("Loading complete!")
    
data_S = np.array(l_S)
data_B = np.array(l_B)


data_S = np.vstack((Branch_Names,data_S))
data_B = np.vstack((Branch_Names,data_B))

wanted_variables =np.array( ["mww","deltar_ljj","deltaphi_ljj","bb_pt","deltaphi_bbljj","deltar_bbljj","ww_pt"])

variables =np.array(data_S[0,:])

ind_s = np.array([])

print("\nSelecting interesting variables")
with tqdm(range(len(wanted_variables))) as pbar:
    for j in range(0,len(wanted_variables)):
        for i in range(0,len(variables)):
            if(variables[i] == wanted_variables[j]):
                ind_s = np.append(ind_s,i)
        pbar.update(1)

ind_s = ind_s.astype(int)

signal = np.ones(len(data_S[:,1]))
bkg = np.zeros(len(data_B[:,0]))
#data_S[:,1]
for i in ind_s:
    signal = np.c_[data_S[:,i],signal]
    bkg = np.c_[data_B[:,i],bkg]
    
np.save("/home/wahid/Desktop/University/Python_Scripts/Data/Signal_var.npy",signal)
np.save("/home/wahid/Desktop/University/Python_Scripts/Data/Background_var.npy",bkg)

signal = signal[1:,:]
bkg = bkg[1:,:]
signal = signal.astype(float)
bkg = bkg.astype(float)


np.save("/home/wahid/Desktop/University/Python_Scripts/Data/Signal.npy",signal)
np.save("/home/wahid/Desktop/University/Python_Scripts/Data/Background.npy",bkg)


