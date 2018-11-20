

using namespace std;

int TMVATrain(){
    
    TMVA::Tools::Instance();
    
    //TTree reading
    
    TString Dataset = "/home/wahid/Scrivania/MLP/Dataset/";
    
    TString Sname = Dataset+"/HH_bis_cut.root" ; //Signal
    
    TFile * signal = TFile::Open(Sname);
        
    TString Bname = Dataset+"/ttbar-semi-lep_cut.root" ; //Background
    
    TFile * background = TFile::Open(Bname);
    
    //TTree loading
    TTree * sig = (TTree*) signal->Get("tree"); 
    TTree * bkg = (TTree*) background->Get("tree");
    
    string n;
    
    cout << "Train number: " << endl; 
    cin >> n;    
    
    TString number = n;
    
    TString outputfileName = ("TMVAtrain"+number+".root");
    
    TFile * outputfile = TFile::Open(outputfileName,"RECREATE");
    
    
    //Factory definition
     
    
    TMVA::Factory * factory = new TMVA::Factory 
    (
      "TMVAClassification", 
      outputfile,                                           
      "!V:!Silent:Color:DrawProgressBar:Transformations=I;P;G:AnalysisType=Classification" 
    ) ;
    
     
    
    //Adding input variables 
    
    
    TMVA::DataLoader * dataloader = new TMVA::DataLoader("NN_classification"+number);
    
    
    dataloader->AddVariable ("mww", 'F');
    dataloader->AddVariable("deltar_ljj", 'F');
    dataloader->AddVariable("deltaphi_ljj", 'F');
    dataloader->AddVariable("bb_pt", 'F');
    dataloader->AddVariable("deltaphi_bbljj", 'F');
    dataloader->AddVariable("deltar_bbljj", 'F');  
    dataloader->AddVariable ("ww_pt", 'F');
    dataloader->AddVariable("Ptm",'F');
    dataloader->AddVariable("deltar_bb",'F');
    dataloader->AddVariable("deltaphi_bb", 'F');
    dataloader->AddVariable("Ht",'F'); 

    //Spectator variables
  //  dataloader->AddSpectator("mbb", 'F');
  // dataloader->AddSpectator("mww", 'F');
  // dataloader->AddSpectator("mvbs", 'F');
    
    //Adding signal and background TTree
    
    dataloader->AddSignalTree(sig, 1.);
    dataloader->AddBackgroundTree(bkg, 1.);


    
    
 
//NN configuration

    TString nt = "20000";
    
    TCut mycuts = ""; 
    TCut mycutb = mycuts;
    
    



      dataloader->PrepareTrainingAndTestTree
    ( 
      mycuts, 
      mycutb,
      "nTrain_Signal="+nt+":nTrain_Background="+nt+":SplitMode=Random:NormMode=EqualNumEvents:!V" ); //:nTest_Signal="+nt+":NTest_Background="+nt+"

      factory->BookMethod 
    (
      dataloader, 
      TMVA::Types::kMLP, 
      "MLP", 
      "!H:!V:NeuronType=tanh:VarTransform=Gauss:NCycles=10:HiddenLayers=1:TestRate=15:TrainingMethod=BP:SamplingTesting=True:ConvergenceImprove=1e-30:ConvergenceTests=75" 
    ) ;
    

      // Train MVAs using the set of training evelaureants
  //factory->OptimizeAllMethodsForClassification 	("ROCIntegral","FitGA");
  factory->TrainAllMethods () ;

  // Evaluate all MVAs using the set of test events
  factory->TestAllMethods () ;

  // Evaluate and compare performance of all configured MVAs
  factory->EvaluateAllMethods () ;
  
  
  
  

  
  //ROC
//   cout << "\n\n ===================> AUC <======================\n\n" << endl;
//   cout << "\t\t\t AUC:" << "\t" << factory->GetROCIntegral(dataloader, "MLP") << endl;
//   cout << "\n\n ==================================================== " << endl;
//     
//  
//   
//   TCanvas* ROC = factory->GetROCCurve(dataloader);
//   ROC -> Draw();
  
  
    //Closing and deleting objects //
  

  factory->Write();
  ROC->Write();
  dataloader->Write();
  
  outputfile->Close () ;

  delete factory ;
  delete dataloader ;
  delete sig;
  delete bkg ;
  delete outputfile ;

  if (!gROOT->IsBatch()) TMVA::TMVAGui( outputfileName );
    
    return 0;
}


int main(int argc, char ** argv){

    return TMVATrain();
    
}
