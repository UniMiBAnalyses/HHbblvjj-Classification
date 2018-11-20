
using namespace std;


int TMVARead(){

    TMVA::Tools::Instance();
    
    //TTree reading 
    
    
    TString Dataset = "/home/wahid/Scrivania/MLP/Dataset/";
    
    TString Sname = Dataset+"/HH_bis.root" ; //Signal
    TString Stotname = Dataset+"HH_bis.root";
    TString Btotname = Dataset+"ttbar-semi-lep.root";//Background
    
    TFile * signal = TFile::Open(Sname);
    TFile * signaltot = TFile::Open(Stotname);
    TFile * bkgtot = TFile::Open(Btotname);
        
    TString Bname = Dataset+"/ttbar-semi-lep_cut.root" ;
    
    TFile * background = TFile::Open(Bname);
    
    TTree * sig = (TTree*) signal->Get("tree");
    TTree * bkg = (TTree*) background->Get("tree");
    TTree * sig_tot = (TTree*) signaltot->Get("tree");
     TTree * bkg_tot = (TTree*) bkgtot->Get("tree");
    string n;
    
    cout << "Train number : \t" ; 
    cin >> n;
    TString number = n; //Numero training
    TString Path = "/home/wahid/Scrivania/MLP/Train_prove/NN_classification"+number+"/"; //Dir training results
    TString ClassFile = Path+"TMVAtrain"+number+".root"; //File risultati training
     
    TFile inputClass(ClassFile);
     //Prendo test e train tree per calcolare accuracy
     
    TDirectoryFile * NN_classification = (TDirectoryFile*) inputClass.Get ( "NN_classification" + number );
    
    TTree* trainTree = (TTree*)NN_classification->Get("TrainTree");
    TTree* testTree = (TTree*)NN_classification->Get("TestTree");
    TMVA::Factory * factory = (TMVA::Factory*)NN_classification->Get("factory");
    TMVA::DataLoader* dataloader = (TMVA::DataLoader*)NN_classification->Get("dataloader");
  
    
        
    
    TString outputfileName = (Path+"TMVAread"+number+".root");
    
    
    
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
  
  
  


reader->AddVariable ("mww", &mww) ; 
reader->AddVariable("deltar_ljj", &deltar_ljj);
reader->AddVariable("deltaphi_ljj", &deltaphi_ljj); 
reader->AddVariable("deltaphi_bbljj", &deltaphi_bbljj);
reader->AddVariable("bb_pt", &bb_pt);

reader->AddVariable("ww_pt",&ww_pt);
reader->AddVariable("n_pt", &n_pt);
reader->AddVariable("Ptm", &Ptm);
reader->AddVariable("deltar_bb", &deltar_bb);
reader->AddVariable("deltaeta_ljj", &deltaeta_ljj);
reader->AddVariable("deltaphi_bb", &deltaphi_bb);
reader->AddVariable("Ht",&Ht);

  

    
    reader->BookMVA("NN", Path+"weights/TMVAClassification_MLP.weights.xml") ;
 

     
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

    
    sig->SetBranchAddress("lep_pt", &lep_pt);
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
    
    sig_tot->SetBranchAddress("ww_pt", &ww_pt);
    sig_tot->SetBranchAddress("Ptm", &Ptm);
    bkg_tot->SetBranchAddress("ww_pt", &ww_pt);
    bkg_tot->SetBranchAddress("Ptm", &Ptm);
    

    
    double binR = 100;
    double maxR = 1.0;


  
  TH1F * h_NN_signal = new TH1F ("signal", "NN discriminant distribution", binR, 0., maxR) ;
  TH1F * h_NN_background = new TH1F ("h_NN_background", "NN discriminant distribution", binR, 0., maxR) ;
  TH1F * sig_norm = new TH1F ("Signal", "Signal probability distribution", binR, 0., maxR) ;
  TH1F * bkg_norm = new TH1F ("Background", "Background probability distribution", binR, 0., maxR) ;
  TH1F * Significance  = new TH1F ("Significance" , "Significance", binR,0,maxR);
  TH1F * Significance2  = new TH1F ("Significance2" , "Significance2", binR,0,maxR);
  TH1F * evS = new TH1F ("S","S", binR,0,maxR);
  TH1F * evB = new TH1F ("B","B", binR,0,maxR);
  TH1F *nevS = new TH1F("NevS","NevS", binR,0,maxR);
  TH1F *nevB = new TH1F("NevB","NevB", binR,0,maxR);
  
  TH2F * plt1 = new TH2F("plot","plot",50,0,200,50,0,200);
  TH2F * plt2 = new TH2F("plt1","plt1", 50,0,200,50,0,200);
  
  double cut = 0;
  double cutmax=0;
  double sig_cut = 0;
  double bkg_cut = 0;
  int nbin = 100;
  double L = 150000; //picobarn Luminosity
  double sigmaS = 0.01997;//picobarn XS_s
  double sigmaB = 137.5;//picobarn XS_b
  double eff_signal = 0.083708; //Preselection eff
  double eff_bkg = 0.220374; //Preselection eff
  double fom = 0 , fom_max  = 0;
  double N_mc = 1000000;
  double fomerr=0, fom_maxerr=0;
  double binmax = 0;
  double fom_af , fom_aferr;
  double nevaf;
  double effsignal;
  
    for (int i = 0 ; i < sig->GetEntries (); ++i)
    {
      sig->GetEntry (i) ;
      
      h_NN_signal->Fill (reader->EvaluateMVA ("NN")) ;
    
 
    }
    
      for (int i = 0 ; i < bkg->GetEntries (); ++i)
    {
      
      bkg->GetEntry (i) ;
      
      h_NN_background->Fill (reader->EvaluateMVA ("NN")) ;
      //cout << i << endl;
      
    }
    
    for(int i = 0; i < 100000; i++){
        
        sig_tot->GetEntry(i);
  
        plt1->Fill(ww_pt,Ptm);   
        
        
    }
    
     for(int i = 0; i <100000; i++){
        
        bkg_tot->GetEntry(i);

        plt2->Fill(ww_pt,Ptm);       
        
    }
    
    
    
    
 //Riempio istogrammi nuovi da normalizzare che utilizzo per calcolare la significatività
   
    for(int j = 1; j <= binR; j++){
			bkg_norm->SetBinContent(j, h_NN_background->GetBinContent(j));
    		sig_norm->SetBinContent(j, h_NN_signal->GetBinContent(j));
		}
		//Scalo gli istogrammi
		
		sig_norm->Scale(((L*sigmaS)/N_mc));
		bkg_norm->Scale(((L*sigmaB)/N_mc));
//         
		 

    //Ciclo dove faccio i tagli scorrendo su ogni bin.
    double bin = 0;
      	for(int j = 1; j <= binR; j++){
            bin ++;
            
			double s_sx = sig_norm->Integral(1,j); //Left integral
            
			double s_dx = sig_norm->Integral(j, binR); //Right integral
			double eff_signal = s_dx/sig_norm->Integral();
            
            double b_sx = bkg_norm->Integral(1,j);
			double b_dx = bkg_norm->Integral(j, binR); //
            double eff_bkg = b_dx/bkg_norm->Integral();
            double b_dx_err = 0.1*0.1*b_dx*b_dx;
			//double purity_s = s_dx*sig_norm->Integral() / (s_dx*sig_norm->Integral()+b_sx*h_NN_background->Integral());
            //cout << "Err:\t" << b_dx_err << endl;;
            nevS->SetBinContent(j,s_dx); //Signal efficiency
            nevB->SetBinContent(j,b_dx); //Background efficiency
            if ( b_dx != 0.){
            double fom = (s_dx) / sqrt((b_dx)); //Significance without error
            double fomerr = (s_dx)/sqrt(b_dx + b_dx_err);//Significance with error (Factor of merith)
            Significance->SetBinContent(j, fom); //
            Significance2->SetBinContent(j,fomerr); //
             
//              
            if(j % 1000 == 0){
                cout << j << endl;
                
            }

              if (fom > fom_max) {
            fom_max = fom; //Scelgo significatività più alta
            binmax = j;
            cutmax = (bin)*maxR/binR;  //Vedo il bin a che taglio corrisponde
        
            
        }

            }
            }
                
            
		}

    
  h_NN_signal->SetFillStyle(3003);
  h_NN_background->SetFillStyle(3007);
  h_NN_signal->SetFillColor(2);
  h_NN_background->SetFillColor(4);
  h_NN_signal->SetLineColor(2) ;
  h_NN_background->SetLineColor(4) ;
  h_NN_signal->SetStats (0) ;
  h_NN_background->SetStats (0) ;
  h_NN_signal->SetXTitle("t");
  h_NN_background->SetXTitle("t");
  
  h_NN_signal->SetYTitle("f(t)");
  h_NN_background->SetYTitle("f(t)");
  
  
 
  TFile * outputfile = TFile::Open(outputfileName,"RECREATE");
  outputfile->cd();
  h_NN_background->Write();
  h_NN_signal->Write();
  
  TApplication * app = new TApplication("app",0,0);

  TCanvas *c1 = new TCanvas("c1","c1",50,50,1000,800) ;

  
  TLegend * legend1 = new TLegend  (0.3,0.7,0.6,0.9);
  
  c1->SetLogy ();
  
  
  if (h_NN_signal->GetMaximum () > h_NN_background->GetMaximum ()) h_NN_background->Draw () ;
  else h_NN_background->Draw () ;
  
  h_NN_signal->Draw () ;
  h_NN_background->Draw ("same") ;
  
  legend1->AddEntry(h_NN_signal,"Signal", "f");
  legend1->AddEntry(h_NN_background,"Background", "f");
  legend1->Draw();
  //c1->Print (Path+"NN_distributions"+number+".pdf", "pdf") ;  
   
  TCanvas * s = new TCanvas("Significance");
  TLegend * legends = new TLegend  (0.3,0.7,0.6,0.9);
  Significance->SetLineWidth(3);
  Significance2->SetLineWidth(3);
  Significance->SetLineColor(kRed);
  Significance2->SetLineColor(kBlue);
  legends->AddEntry(Significance, "No error");
  legends->AddEntry(Significance2, "With error");
  Significance->SetXTitle("t");
  Significance->SetYTitle("Significance");
  Significance->SetStats(0);
  Significance2->SetStats(0);
  Significance->Draw("");
  legends->Draw("same");
  Significance2->Draw("same");
  //s -> Print(Path+"Significance.pdf" , "pdf");
  
  //Kolmogorov test
  
  
  TH1F * sig_train = new TH1F ("Sig_train", "NN discriminant distribution", 100, 0., 1) ;
  TH1F * bkg_train = new TH1F ("bkg_train", "NN discriminant distribution", 100, 0., 1) ;
  
  TH1F * sig_test = new TH1F ("sig_test", "NN discriminant distribution", 100, 0., 1) ;
  TH1F * bkg_test = new TH1F ("bkg_test", "NN discriminant distribution", 100, 0., 1) ;
  
  
  
    for (int i = 0 ; i < trainTree->GetEntries (); ++i)
    {
      trainTree->GetEntry (i) ;
      sig_train->Fill (reader->EvaluateMVA ("NN")) ;
 
    }
    
      for (int i = 0 ; i < testTree->GetEntries (); ++i)
    {
      testTree->GetEntry (i) ;
      sig_test->Fill (reader->EvaluateMVA ("NN")) ;
      //cout << i << endl;
      
    }

    
//Results plot

 TCanvas  c2;
  
  TLegend * legend2 = new TLegend  (0.3,0.7,0.6,0.9);  
  evS->SetLineColor(kRed);
  evB->SetLineColor(kBlue);
  evS->SetStats(0);
  evB->SetStats(0);
  evB->SetLineWidth(3);
  evS->SetLineWidth(3);
  evS->SetXTitle("t");
  evS->SetTitle("Signal and background efficiencies");
  evS->SetYTitle("Efficiency");
  c2.SetLogy();
  legend2->AddEntry(evS, "Signal efficiency");
  legend2->AddEntry(evB, "Background efficiency");  
  evS->Draw("histo");
  evB->Draw("histo same");
  legend2->Draw("same");
  
  TCanvas  c3;
  
  c3.Divide(3,1);
  
  c3.cd(1);
  Significance->SetLineColor(kRed);
  Significance->SetLineWidth(3);
  Significance->SetXTitle("t");
  Significance->SetYTitle("Significance");
  Significance->SetStats(0);
  Significance->SetTitle("Significance 3000fb^{-1}");
  Significance->Draw("histo");
  
  c3.cd(2);
  nevS->SetLineColor(kRed);
  nevS->SetLineWidth(3);
  nevS->SetXTitle("t");
  nevS->SetYTitle("# Events");
  nevS->SetTitle("Number of signal events vs t_{cut}");
  nevS->SetStats(0);
  nevS->Draw("histo");
  
  c3.cd(3);
  nevB->SetLineColor(kBlue);
  nevB->SetLineWidth(3);
  nevB->SetXTitle("t");
  nevB->SetYTitle("# Events");
  nevB->SetTitle("Number of background events vs t_{cut}");
  nevB->SetStats(0);
  nevB->Draw("histo");
  
    
    TCanvas c4;
  
  TLegend * legend4 = new TLegend  (0.3,0.7,0.6,0.9);  
  Significance2->SetLineColor(kBlue);
  Significance2->SetLineWidth(3);
  Significance2->SetXTitle("t");
  Significance2->SetYTitle("Significance");
  Significance2->SetStats(0);
  Significance2->SetTitle("Significance 3000fb^{-1}, with systematic error");
  Significance2->Draw("histo");
  
TCanvas c5;
  
  TLegend * legend5 = new TLegend  (0.3,0.7,0.6,0.9);  
  Significance->SetLineColor(kRed);
  Significance->SetLineWidth(3);
  Significance->SetXTitle("t");
  Significance->SetYTitle("Significance");
  Significance->SetStats(0);
  Significance->SetTitle("Significance 3000fb^{-1}, without systematic error");
  Significance->Draw("histo");file:///home/wahid/Scrivania/TESI/abstract.pdf
  
        
        

  //cout << "Kol test:" << sig_train->KolmogorovTest(sig_test) << endl;
//  TCanvas * sov = new TCanvas("MET vs P_{t,WW}");
//  plt1->SetTitle("MET vs P_{t,WW}");
//  plt1->SetStats(0);
//  TLegend * l1 = new TLegend(0.3,0.7,0.6,0.9);
//  plt1->SetXTitle("P_{t,WW}");
//  plt1->SetYTitle("MET");
// //  plt1->SetFillStyle(3019);
// //  plt2->SetFillStyle(3644);
//  plt1->SetFillColor(kRed);
//  plt2->SetFillColor(kBlue);
//  
//  TExec * ex1 = new TExec("ex1","gStyle->SetPalette(kDarkBodyRadiator);");
//  TExec * ex2 = new TExec("ex2","gStyle->SetPalette(kDeepSea);");
//  
//  
//  l1->AddEntry(plt1,"Signal");
//  l1->AddEntry(plt2,"Background");
//  
//  plt1->Draw("CONT1");
//  ex1->Draw("");
//  ex2->Draw("same");
//  plt2->Draw("CONT same");
//  l1->Draw("same");   
//   app->Run();
  
  
  s->Write();
  c1->Close();
  s->Close();
  c2.Close();
  outputfile->Close();

  


  delete reader;
    return 0;
    
}
