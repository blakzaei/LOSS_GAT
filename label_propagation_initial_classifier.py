#-- IMPORT --------------------------------------------------------------------

from util import util, gat_util
import gat

import torch

import pandas as pd
import numpy as np

import copy
###############################################################################


#-- Class LP_Clssifier: Label Propagation using an Initial Classifier ---------
class LP_Classifier():
    
    def __init__(self , k , alpha, labeled_percent, iteration, ds_title, epoch): 
        
        #-- log --
        print('Start Label Propagation Using an Initial Classifier ----------')
        print('\titeration=%d - k=%d - alpha = %f' \
                  %(iteration,k,alpha))
        
        #-- initlize params ---------------------------------------------------
        self.k = k
        self.alpha = alpha           
        self.labeled_percent = labeled_percent
        self.iteration = iteration
        self.ds_title = ds_title    
        self.epoch = epoch
        
        
        self.input_path = 'results/' 
        
        self.ids_file = self.input_path + 'doc2vec/ids.npy'
        self.doc2vec_file = self.input_path + 'doc2vec/doc2vec_vect.csv'
        self.true_labels_file = self.input_path + 'doc2vec/labels.npy'
        
        self.graph_file = self.input_path + 'graph/knn_graph_' + str(self.k) + '.graphml'
        
        self.rp_katz_file = self.input_path +'lp_katz/' + str(self.iteration) + \
            '_' + str(self.labeled_percent) + '_rp_katz_' +  str(self.k) + \
                '_' + str(self.alpha) + '.text'            
        self.rn_katz_file = self.input_path +'lp_katz/' + str(self.iteration) + \
            '_' + str(self.labeled_percent) + '_rn_katz_' +  str(self.k) + \
                '_' + str(self.alpha) + '.text'  
        
        self.labeled_indexes_file = self.input_path  + 'split/' + str(self.iteration) + \
            '_' + str(self.labeled_percent) + '_labeled_indexes.npy'       
        
        
        self.output_path = 'results/lp_classifier/'
        util.create_folder(self.output_path)      
        
        self.model_file = self.output_path + str(self.iteration) \
            + '_' + str(self.labeled_percent) +'_model_' + str(self.k) \
                + '_' + str(self.alpha) +  '.pt'
        self.predicted_labels_file = self.output_path + str(self.iteration) \
            + '_' + str(self.labeled_percent) +'_predicted_' + str(self.k) \
                + '_' + str(self.alpha) +  '.pt'
        
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('\tdevice:' , self.device)
        
        self.labeled_indexes = np.load(self.labeled_indexes_file)       
        
          
        self.X = gat_util.create_X(self.doc2vec_file)        
        self.true_labels = gat_util.load_true_labels(self.true_labels_file)
        
        self.edge_index, _ = gat_util.create_edge_indexes(self.graph_file, self.doc2vec_file)        
        self.rp_katz , self.rn_katz = self.create_rp_and_rn()
        
        #-- Run ---------------------------------------------------------------      
        self.run()
        
        #-- log --
        print('Finish Label Propagation Using an Initial Classifier ---------')           
    
        
    
    #--------------------------------------------------------------------------
    def create_rp_and_rn(self):
        
        #-- log --
        print('\tSaving RP_Katz and RN_Katz as indexes of samples ... ')
        
        #true_labels = np.load(self.true_labels_file)
        real_indexes = np.load(self.ids_file)
        
        real_indexes = list(real_indexes)
        
        #print('real_indexes:' , len(real_indexes))
        #print(true_labels)
        #print(real_indexes)                    
                    
                    
        f_rp = open(self.rp_katz_file, "r")
        f_rn = open(self.rn_katz_file, "r")
        
        #labeled_indexes = []
        rp_indexes = []
        rn_indexes = []
        
        for line in f_rp:
            #id , label = line.split('\t')
            #id , tag = line.split(':')
            
            id = line.strip()
        
            i = real_indexes.index(id)
        
            #labeled_indexes.append(i)
            rp_indexes.append(i)
        
        for line in f_rn:
            #id , label = line.split('\t')
            #id , tag = id.split(':')
            
            id = line.strip()
        
            i = real_indexes.index(id)
        
            #labeled_indexes.append(i)
            rn_indexes.append(i)
        
        
        
        
        rp_file = self.output_path + str(self.iteration) + '_' + str(self.labeled_percent) \
            + '_rp_' + str(self.k) + '_' + str(self.alpha)
            
        rn_file = self.output_path + str(self.iteration) + '_' + str(self.labeled_percent) \
            + '_rn_' + str(self.k) + '_' + str(self.alpha)
                    
        np.save(rp_file , rp_indexes)
        np.save(rn_file, rn_indexes)
        
        print('\tRP: %d , RN:%d' %(len(rp_indexes), len(rn_indexes)))
        
        return rp_indexes , rn_indexes
    #--------------------------------------------------------------------------       
    
    
    
    def run(self):          
        
        self.edge_index = self.edge_index.t().contiguous()
        print('\tedge_index:', self.edge_index.shape)
        
        y_train , labeled_indexes = gat_util.create_y_semi_supervised(
            self.true_labels_file,
            self.rp_katz,
            self.rn_katz)
        y_train = y_train.to(self.device)
        print('\tsemi-supervised y_train:' , y_train.shape)        

        #-- Create a Binary train_mask for just labeled samples--
        n_nodes = self.true_labels.shape[0]         
        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        train_mask[labeled_indexes] = True
        #print('\ttrain_mask: ' , torch.sum(train_mask).item())

        gat2 = gat.GAT_Network(num_features= self.X.size(-1),
                               num_hidden= 512,
                               num_outs= 1,
                               num_heads=4).to(self.device)      
        
        
        optimizer = torch.optim.Adam(gat2.parameters(),
                                     lr=0.01,
                                     weight_decay=5e-7)

        train_losses = []
        test_losses = []
        test_accs = []

        best_macro = 0
        best_epoch = 0
        best_model = None        
        best_labels = None       

        for epoch in range(1, self.epoch+1):    
            
            loss_train, embeddings = gat_util.train(model= gat2,
                                               optimizer= optimizer,
                                               device= self.device,
                                               x= self.X,
                                               edge_index= self.edge_index,
                                               y_train = y_train,
                                               train_mask= train_mask)
            
            acc_test , loss_test, macro_f1 , predictions = gat_util.test(model= gat2,
                                                              device= self.device,
                                                              x= self.X,
                                                              edge_index= self.edge_index,
                                                              y_train= y_train,
                                                              y_true= self.true_labels,
                                                              labeled_indexes= self.labeled_indexes)
            if epoch==1 or epoch%10 ==0 or epoch==self.epoch:
                print(f'\tEpoch {epoch:03d}: Loss: {loss_train:.4f} | Unlabeled Acc: {acc_test:.4f} | Macro-f1: {macro_f1:.4f}')

            train_losses.append(loss_train)
            test_accs.append(acc_test)
            test_losses.append(loss_test)

            if macro_f1> best_macro:
                best_macro = macro_f1
                best_epoch = epoch
                best_model = copy.deepcopy(gat2)                
                best_labels = predictions             



        #-- log --        
        print('\tFinish Train :)')     
        print('\tSaving Results ...')        
        
        torch.save(predictions, self.predicted_labels_file)   
        torch.save(best_model.state_dict(), self.model_file)


        predicted_fake = torch.nonzero(best_labels == 1).squeeze()
        predicted_real = torch.nonzero(best_labels == 0).squeeze()

        #print('predicted_fake:' , predicted_fake.shape)
        #print('predicted_real:' , predicted_real.shape)

        

        
        
        
        


    

    


    
    
    






################################################################################



    

    

           







