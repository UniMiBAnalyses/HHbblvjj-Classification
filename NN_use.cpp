// c++ -g -o NN_use NN_use.cpp `root-config --libs --glibs --cflags` -lTMVA -lTMVAGui
#include <iostream>
#include <sstream>
#include <string>
#include <fstream>
#include <algorithm>
#include <vector>

#include "TNtuple.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TRandom.h"
#include "TCanvas.h"
#include "TMVA/Factory.h"
#include "TMVA/DataLoader.h"
#include "TMVA/Tools.h"
#include "TMVA/Reader.h"
#include "TMVA/TMVAGui.h"
#include "TCut.h"
#include "TF1.h"
#include "TApplication.h"
#include "TLatex.h"
#include "TColor.h"
#include "TString.h"

using namespace std ;

// a more complete example from the ROOT website can be found here:
// https://root.cern.ch/doc/v608/TMVAMultipleBackgroundExample_8C.html



int main (int argc, char ** argv)
{
  
      /* if (argc < 4){
        cout << "Usage: " << argv[0] << " sigbkg.root " << endl;
        return 1;
    } */
    
    string n;
    string method;
    
//     cout << "Inserire numero training : \t" ;
//     cin >> n;

    

    
    TString  numberCl = "7X"; //Number of the classification test
    TString Dataset = "/home/wahid/Scrivania/MLP/Dataset/"; //Path of the data
    TString Path = "/home/wahid/Scrivania/MLP/Train_prove/NN_classification"+numberCl+"/";
    
 
    double cut = 0.99989;
    //cout << "Inserire valore del taglio: \t";
    //cin >> cut;
    //Loading the root files //
    
    
    TFile * input = TFile::Open(Dataset+"/dataset_nuovo.root");
    TFile * sinput = TFile::Open(Dataset+"/HH_bis_nuovo.root");
    TFile * binput = TFile::Open(Dataset+"/ttbar-semi-lep_nuovo.root");
    
    
    //Loading the tree
    TTree * dataset  = (TTree*)input->Get("tree");
    TTree * sig  = (TTree*)sinput->Get("tree");
    TTree * bkg  = (TTree*)binput->Get("tree");
    
TString outputfileName = ("TMVAuse"+numberCl+".root");
    
TFile * outputfile = TFile::Open(outputfileName,"RECREATE");
    
    
    TMVA::Reader * reader = new TMVA::Reader( "!Color:!Silent" );
    
 //Defining variables
    
    float lep_eta, lep_pt, lep_E, lep_Et, lep_phi;
    float n_eta, n_pt;
    float j1_eta, j1_pt, j1_phi, j2_eta, j2_pt, j2_phi;
	float b1_eta, b1_pt, b1_phi, b2_eta, b2_pt, b2_phi;
    float mww, mjj, mbb;
	float ww_pt, j_pt, bb_pt;
	float j_Et, bb_Et, ww_Et;
	float Ht, Htnu, Ptm;
	float deltar_ljj, deltar_bbljj, deltar_bb;
	float deltaphi_ljj, deltaphi_bbljj, deltaphi_bb;
	float deltaeta_ljj, deltaeta_bbljj, deltaeta_bb; 
  
  //Reading variables

     reader->AddVariable ("mww", &mww);
     reader->AddVariable("deltar_ljj", &deltar_ljj);
     reader->AddVariable("deltaphi_ljj", &deltaphi_ljj);
     reader->AddVariable("deltaphi_bbljj", &deltaphi_bbljj);
     reader->AddVariable("bb_pt", &bb_pt);
     reader->AddVariable("ww_pt", &ww_pt);
     reader->AddVariable("n_pt", &n_pt);
     reader->AddVariable("Ptm",&Ptm);
     reader->AddVariable("deltar_bb", &deltar_bb);
     reader->AddVariable("deltaeta_ljj",&deltaeta_ljj);
     reader->AddVariable("deltaphi_bb", &deltaphi_bb);
     reader->AddVariable("Ht",&Ht);

  
  

    reader->BookMVA("NN", Path + "weights/TMVAClassification_MLP.weights.xml") ; //Dir train results
  
     
    sig->SetBranchAddress("lep_pt", &lep_pt);
	sig->SetBranchAddress("lep_eta", &lep_eta);
	sig->SetBranchAddress("lep_E", &lep_E);
	sig->SetBranchAddress("lep_Et", &lep_Et);
    sig->SetBranchAddress("n_pt", &n_pt);
    sig->SetBranchAddress("j1_pt", &j1_pt);
	sig->SetBranchAddress("j1_eta", &j1_eta);
    sig->SetBranchAddress("j2_eta", &j2_eta);
    sig->SetBranchAddress("j2_pt", &j2_pt);	
    sig->SetBranchAddress("b1_pt", &b1_pt);
	sig->SetBranchAddress("b1_eta", &b1_eta);
    sig->SetBranchAddress("b2_pt", &b2_pt);

	sig->SetBranchAddress("mjj", &mjj);
	sig->SetBranchAddress("j_pt", &j_pt);
	sig->SetBranchAddress("j_Et", &j_Et);
	sig->SetBranchAddress("mww", &mww);
	sig->SetBranchAddress("ww_pt", &ww_pt);
	sig->SetBranchAddress("ww_Et", &ww_Et);
	sig->SetBranchAddress("mbb", &mbb);
	sig->SetBranchAddress("bb_pt", &bb_pt);
	sig->SetBranchAddress("bb_Et", &bb_Et);
	
	sig->SetBranchAddress("Ht", &Ht);
	sig->SetBranchAddress("Htnu", &Htnu);
	sig->SetBranchAddress("Ptm", &Ptm);

	sig->SetBranchAddress("deltaphi_bb", &deltaphi_bb);
	sig->SetBranchAddress("deltaphi_ljj", &deltaphi_ljj);
	sig->SetBranchAddress("deltaphi_bbljj", &deltaphi_bbljj);

	sig->SetBranchAddress("deltaeta_bb", &deltaeta_bb);
	sig->SetBranchAddress("deltaeta_ljj", &deltaeta_ljj);
	sig->SetBranchAddress("deltaeta_bbljj", &deltaeta_bbljj);

	sig->SetBranchAddress("deltar_bb", &deltar_bb);
	sig->SetBranchAddress("deltar_ljj", &deltar_ljj);
	sig->SetBranchAddress("deltar_bbljj", &deltar_bbljj);

    
    bkg->SetBranchAddress("lep_pt", &lep_pt);
	bkg->SetBranchAddress("lep_eta", &lep_eta);
	bkg->SetBranchAddress("lep_E", &lep_E);
	bkg->SetBranchAddress("lep_Et", &lep_Et);
    bkg->SetBranchAddress("n_pt", &n_pt);
    bkg->SetBranchAddress("j1_pt", &j1_pt);
	bkg->SetBranchAddress("j1_eta", &j1_eta);
    bkg->SetBranchAddress("j2_eta", &j2_eta);
    bkg->SetBranchAddress("j2_pt", &j2_pt);	
    bkg->SetBranchAddress("b1_pt", &b1_pt);
	bkg->SetBranchAddress("b1_eta", &b1_eta);
    bkg->SetBranchAddress("b2_pt", &b2_pt);

	bkg->SetBranchAddress("mjj", &mjj);
	bkg->SetBranchAddress("j_pt", &j_pt);
	bkg->SetBranchAddress("j_Et", &j_Et);
	bkg->SetBranchAddress("mww", &mww);
	bkg->SetBranchAddress("ww_pt", &ww_pt);
	bkg->SetBranchAddress("ww_Et", &ww_Et);
	bkg->SetBranchAddress("mbb", &mbb);
	bkg->SetBranchAddress("bb_pt", &bb_pt);
	bkg->SetBranchAddress("bb_Et", &bb_Et);
	
	bkg->SetBranchAddress("Ht", &Ht);
	bkg->SetBranchAddress("Htnu", &Htnu);
	bkg->SetBranchAddress("Ptm", &Ptm);

	bkg->SetBranchAddress("deltaphi_bb", &deltaphi_bb);
	bkg->SetBranchAddress("deltaphi_ljj", &deltaphi_ljj);
	bkg->SetBranchAddress("deltaphi_bbljj", &deltaphi_bbljj);

	bkg->SetBranchAddress("deltaeta_bb", &deltaeta_bb);
	bkg->SetBranchAddress("deltaeta_ljj", &deltaeta_ljj);
	bkg->SetBranchAddress("deltaeta_bbljj", &deltaeta_bbljj);

	bkg->SetBranchAddress("deltar_bb", &deltar_bb);
	bkg->SetBranchAddress("deltar_ljj", &deltar_ljj);
	bkg->SetBranchAddress("deltar_bbljj", &deltar_bbljj);
    
    
         
    dataset->SetBranchAddress("lep_pt", &lep_pt);
	dataset->SetBranchAddress("lep_eta", &lep_eta);
	dataset->SetBranchAddress("lep_E", &lep_E);
	dataset->SetBranchAddress("lep_Et", &lep_Et);
    dataset->SetBranchAddress("n_pt", &n_pt);
    dataset->SetBranchAddress("j1_pt", &j1_pt);
	dataset->SetBranchAddress("j1_eta", &j1_eta);
    dataset->SetBranchAddress("j2_eta", &j2_eta);
    dataset->SetBranchAddress("j2_pt", &j2_pt);	
    dataset->SetBranchAddress("b1_pt", &b1_pt);
	dataset->SetBranchAddress("b1_eta", &b1_eta);
    dataset->SetBranchAddress("b2_pt", &b2_pt);

	dataset->SetBranchAddress("mjj", &mjj);
	dataset->SetBranchAddress("j_pt", &j_pt);
	dataset->SetBranchAddress("j_Et", &j_Et);
	dataset->SetBranchAddress("mww", &mww);
	dataset->SetBranchAddress("ww_pt", &ww_pt);
	dataset->SetBranchAddress("ww_Et", &ww_Et);
	dataset->SetBranchAddress("mbb", &mbb);
	dataset->SetBranchAddress("bb_pt", &bb_pt);
	dataset->SetBranchAddress("bb_Et", &bb_Et);
	
	dataset->SetBranchAddress("Ht", &Ht);
	dataset->SetBranchAddress("Htnu", &Htnu);
	dataset->SetBranchAddress("Ptm", &Ptm);

	dataset->SetBranchAddress("deltaphi_bb", &deltaphi_bb);
	dataset->SetBranchAddress("deltaphi_ljj", &deltaphi_ljj);
	dataset->SetBranchAddress("deltaphi_bbljj", &deltaphi_bbljj);

	dataset->SetBranchAddress("deltaeta_bb", &deltaeta_bb);
	dataset->SetBranchAddress("deltaeta_ljj", &deltaeta_ljj);
	dataset->SetBranchAddress("deltaeta_bbljj", &deltaeta_bbljj);

	dataset->SetBranchAddress("deltar_bb", &deltar_bb);
	dataset->SetBranchAddress("deltar_ljj", &deltar_ljj);
	dataset->SetBranchAddress("deltar_bbljj", &deltar_bbljj);
    
    int binmin =0;
    int binmax =700;
    int nbin = 100;
    

    
    
  TH1F * signal = new TH1F ("Signal_mvbs", "Distribution mvbs", nbin,binmin,binmax);
  TH1F * background= new TH1F ("Background_mvbs", "Sovrapposizione fondo vero/fondo NN  mvbs",nbin,binmin,binmax);
 
  TH1F * signal0 = new TH1F ("Signal_mww", "Distribution mww", 100, 0, 400);
  TH1F * bkg0 = new TH1F ("Background_mww", "Distribution mww", 100, 0, 400);
  
  
  TH1F * cutS = new TH1F ("mvbs_NN", "Sovrapposizione segnale vero/segnale NN  mvbs", nbin,binmin,binmax);
  TH1F * cutB = new TH1F ("mvbs_bkg", "Disitribuzioni dopo NN", 20, binmin , binmax);


 TH1F * signal3 = new TH1F ("mwwS", "Separation", 100, 0, 400);
 TH1F * bkg3 = new TH1F ("mww", "Separation mww", 100, 0, 400);
  
 
 TH1F * diff_sig = new TH1F("Diff_sig", "Differenza eventi segnale veri e eventi bkgNN", 100, -1,1);
 TH1F * diff_bkg = new TH1F("Diff_bkg", "Differenza eventi bkg veri e eventi bkg NN", 100, -1,1);
  
 TH2F * plt = new TH2F("plt","plt",200,0,400,200,0,400);
  
  background->SetTitle("b-#bar{b} transverse momentum");
  cutS->SetTitle("b-#bar{b} transverse momentum");
  cutS->SetXTitle("GeV/c^{2}");
  background->SetXTitle("GeV/c^{2}");
  cutS->SetStats(0);


//Riempio istogrammi di segnale e di fondo dati dalla rete  
 
    for(int i = 0; i < sig->GetEntries(); i++){
        
        sig->GetEntry(i);
        
        signal->Fill(bb_pt);
        
    }
    
        for(int i = 0; i < bkg->GetEntries(); i++){
        
        bkg->GetEntry(i);
        
        background->Fill(bb_pt);
        
    }
        
    for (int i = 0 ; i < sig->GetEntries (); ++i)
        
    {
      sig->GetEntry (i) ;
      
      if (reader->EvaluateMVA ("NN") > cut ) 
          
        {      
          
          cutS->Fill (bb_pt);
          
        }
    }
        
    for(int i = 0; i < bkg->GetEntries(); i++)
        
    {
        
        bkg->GetEntry(i);
        
    if(reader->EvaluateMVA("NN") > cut)
    
    { 
        cutB->Fill(bb_pt);
        
    }
    
    }
    

    double intS = signal->Integral();
    double intB = background -> Integral();
    
    double kNorm = intS/intB; //Normalization factor
    
    double  effS = cutS->Integral()/signal->Integral();
    double  effB = cutB->Integral()/background->Integral();    
    
    
    cout << "\n ============================ " << endl;
    cout << "SIGNAL EFFICIENCY: \t " << effS << endl;
    cout << "================================ " << endl;
    
    
        cout << "\n ============================ " << endl;
    cout << "BACKGROUND EFFICIENCY: \t " << effB << endl;
    cout << "================================ " << endl;
    
    cutB->Scale(kNorm*200);
    background->Scale(kNorm);   

    
    
    
TApplication * app = new TApplication("app",0,0);



TCanvas * c1 = new TCanvas("c1","c1");     

c1->SetLogy();

c1->Divide(2,1);

c1->cd(1);

    
    
signal->SetFillStyle(3003);
background->SetFillStyle(3003);
signal->SetFillColor(kOrange+7);
background->SetFillColor(kAzure-6);
signal->SetLineColor(kOrange+7);
background->SetLineColor(kAzure-6);

cutS->SetFillStyle(3004);
cutB->SetFillStyle(3004);
cutS->SetFillColor(kRed-3);
cutB->SetFillColor(kOrange-2);
cutS->SetLineColor(kRed-3);
cutB->SetLineColor(kOrange-2);


signal->SetLineWidth(3);
background->SetLineWidth(3);
cutS->SetLineWidth(3);
cutB->SetLineWidth(3);



signal->SetStats(0);

signal->SetYTitle("Norm");


background->SetStats(0);
background->Draw("histo ");
signal->Draw("histo same    ");
TLegend * legend1 = new TLegend  (0.6,0.4,0.9,0.6);

legend1->AddEntry(signal, "Signal");
legend1->AddEntry(cutS, "Signal-Cut");
legend1->AddEntry(background, "Background");
legend1->AddEntry(cutB, "Background-Cut x 200");
legend1->SetTextSize(0.03);



c1->cd(2);
cutS->Draw("histo ");
cutB->Draw("histo same");
legend1->Draw("same");

string s = "Signal efficiency :" ;


TCanvas * plot = new TCanvas("plt");
plt->Draw("");


app->Run();



outputfile->Close();


  
    delete reader ;
    
  return 0 ;
}

  
 


