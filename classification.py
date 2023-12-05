#-- IMPORTS -------------------------------------------------------------------
###############################################################################
from util import util , gat_util
import gat

import torch

import numpy as np
import pandas as pd
import copy
import random
##############################################################################


class Classification():
    def __init__(self , k , alpha, labeled_percent, iteration, ds_title, epoch,
                 aa_thresholds = [float(i) / 10 for i in range(1, 7)],
                 lp_percent =  [0.5 , 0.6, 0.7]): 
        
        #-- log --
        print('Start Clasification ------------------------------------------')
        print('\titeration=%d - k=%d - alpha = %f' \
                  %(iteration,k,alpha))
        
        #-- initlize params ---------------------------------------------------
        self.k = k
        self.alpha = alpha           
        self.labeled_percent = labeled_percent
        self.iteration = iteration
        self.ds_title = ds_title    
        self.epoch = epoch        
        self.aa_thresholds = aa_thresholds
        self.lp_percent  = lp_percent       
                
        
        self.input_path = 'results/' 
        
        self.indexes_file = self.input_path + 'doc2vec/indexes.npy'
        self.doc2vec_file = self.input_path + 'doc2vec/doc2vec_vect.csv'
        self.true_labels_file = self.input_path + 'doc2vec/labels.npy'
        
        self.graph_file = self.input_path + 'sa/aa_knn_graph_' + str(self.k) + '_'  
        
        self.predicted_labels_file = self.input_path + 'lp_classifier/' + \
            str(self.iteration) + '_' + str(self.labeled_percent) + \
                '_predicted_' + str(self.k) + '_' + str(self.alpha) +  '.pt'       
        
        self.labeled_indexes_file = self.input_path  + 'split/' + str(self.iteration) + \
            '_' + str(self.labeled_percent) + '_labeled_indexes.npy' 
            
        self.edge_index_file = self.input_path + 'sa/aa_edge_index_' + \
                         str(self.k) + '_'         
        self.weight_index_file = self.input_path + 'sa/aa_weight_index_' + \
                         str(self.k) + '_'
        
        
        self.output_path = 'results/classification/'
        util.create_folder(self.output_path)
        
        self.results_file = self.output_path + 'classification_results.csv'
        self.model_file = self.output_path + str(self.iteration) \
            + '_' + str(self.labeled_percent) +'_model_' + str(self.k) \
                + '_' + str(self.alpha) +  '.pt'
        
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('\tdevice:' , self.device)
        
        util.create_empty_df_for_results(self.results_file)
        
        self.edge_index = None
        self.weight_index = None
        self.w_matrix = None
        self.sampled_edges = None
        self.sampled_weights = None
        
        self.labeled_indexes = np.load(self.labeled_indexes_file)    
        self.true_labels = gat_util.load_true_labels(self.true_labels_file)
        self.X = gat_util.create_X(self.doc2vec_file)    
        
        
        #-- Run ---------------------------------------------------------------       
        self.run()
        
        #-- log --
        print('Finish Classification ----------------------------------------')   
        
        
        
        
    
    #--------------------------------------------------------------------------    
    def create_rp_and_rn(self , p):   
            
            #-- log --
            print('\tCreating rp and rn by selection %d percent of predicted by initial classifier' %int(p*100))
                
            predicted_labels = torch.load(self.predicted_labels_file,
                                          map_location=torch.device('cpu'))

            #print('predicted_labels:' , predicted_labels.shape)
            #print(predicted_labels)
            #print('...............................')

            #0 = Rreal and 1 =fake
            fake_index = torch.nonzero(predicted_labels == 1).squeeze()
            real_index = torch.nonzero(predicted_labels == 0).squeeze()

            #print('fake_index:' , fake_index.shape)
            #print(fake_index)
            #print('real_index:' , real_index.shape)
            #print(real_index)
            #print('...............................')

            predicted_fake = torch.tensor([idx for idx in fake_index if idx.item() not in self.labeled_indexes])
            predicted_real = torch.tensor([idx for idx in real_index if idx.item() not in self.labeled_indexes])

            #print('predicted_fake:' , predicted_fake.shape)
            #print(predicted_fake)

            #print('predicted_real:' , predicted_real.shape)
            #print(predicted_real)

            #print('...............................')

            #for p in self.lp_percent:
            #print('\n\npercentage: %f -----------------------------' %(p))
            num_samples_fake = int(len(predicted_fake) * p)
            #print('num_samples_fake:' , num_samples_fake)

            sampled_indices = torch.randperm(len(predicted_fake))[:num_samples_fake]
            sampled_fake = predicted_fake[sampled_indices]

            #print('sampled_fake:' , sampled_fake.shape)
            #print(sampled_fake)

            num_samples_real = int(len(predicted_real) * p)
            #print('num_samples_real:' , num_samples_real)
            sampled_indices = torch.randperm(len(predicted_real))[:num_samples_real]
            sampled_real = predicted_real[sampled_indices]

            #print('sampled_real:' , sampled_real.shape)
            #print(sampled_fake)

            rp = self.labeled_indexes.tolist() + sampled_fake.tolist()
            print('\t\trp:', len(rp))

            rn = sampled_real.tolist()
            print('\t\trn:', len(rn))
            
            return rp , rn   
    
    
    #--------------------------------------------------------------------------
    def Convert_G_to_Matrix(self , n_nodes, edge_index, weight_index):
        
        #-- log --
        print('\tConverting Graph to Weight_Matrix ...')
        
        W_Matrix = np.zeros((n_nodes,n_nodes))
    
        for i in range(edge_index.shape[1]):
            u, v = edge_index[:, i]
            w = weight_index[i]
    
            u = int(u.item())
            v = int(v.item())
            w = float(w.item())
            W_Matrix[u,v] = w
            W_Matrix[v,u] = w
    
        return W_Matrix
    #--------------------------------------------------------------------------    
    
    
    #--------------------------------------------------------------------------
    def Sample_Neighbors(self, node, N):
    
        neighbors_index = np.nonzero(self.w_matrix[node])[0]
    
        num_neighbors = np.count_nonzero(self.w_matrix[node])
        #num_sample = int(num_neighbors * PERCENT)
        if num_neighbors < N:
            num_sample = num_neighbors
        else:
            num_sample = N
    
        sum_weights = np.sum(self.w_matrix[node])
        weights = self.w_matrix[node] / sum_weights
        weights_inx = list(np.nonzero(weights))
        weights = weights[tuple(weights_inx)]
    
        selected_indices = random.choices(neighbors_index, weights=weights, k=num_sample)
    
        selected_neighbors = selected_indices
        selected_neighbors_weight = self.w_matrix[node, selected_neighbors]
        selected_neighbors_weight = selected_neighbors_weight.tolist()
    
        for i in range(len(selected_neighbors)):
            e1 = [node , selected_neighbors[i]]
            e2 = [selected_neighbors[i] , node]
            w = selected_neighbors_weight[i]           
    
            self.sampled_edges.append(e1)
            self.sampled_weights.append(w)
    
            self.sampled_edges.append(e2)
            self.sampled_weights.append(w)   
    
        return
    #--------------------------------------------------------------------------
    
    
    #--------------------------------------------------------------------------
    def Samplig(self, N):
        n_nodes = self.X.shape[0]
    
        for node in range(n_nodes):            
            self.Sample_Neighbors(node,N)
    #--------------------------------------------------------------------------   
    
    
    #--------------------------------------------------------------------------
    def run(self):        
                
        for t in self.aa_thresholds:
            for p in self.lp_percent:
                print('\tt=%f - p=%f ----------------------------' %(t,p))      
                
                graph_file = self.graph_file + str(t) + '.graphml'
                
                
                self.edge_index , self.weight_index = gat_util.create_edge_indexes(graph_file,
                                                                                   self.doc2vec_file,
                                                                                   True)       
                                
                main_edge_index = self.edge_index.t().contiguous()
                print('edge_index:', main_edge_index.shape)
                
                main_weight_index = self.weight_index
                print('weight_index:', main_weight_index.shape)
                                

                #-- save graph as dict of neighbors
                n_nodes = self.X.shape[0]
                self.w_matrix = self.Convert_G_to_Matrix(n_nodes,
                                                         main_edge_index,
                                                         main_weight_index)
                
                print('w_matrix:' , self.w_matrix.shape)
                
                rp_indexes , rn_indexes = self.create_rp_and_rn(p)              
                
                
                y_train , labeled_indexes = gat_util.create_y_semi_supervised(
                    self.true_labels_file,
                    rp_indexes,
                    rn_indexes)
                y_train = y_train.to(self.device)
                print('\tsemi-supervised y_train:' , y_train.shape)                  

                #-- Create a Binary train_mask for just labeled samples--
                n_nodes = self.true_labels.shape[0]         
                train_mask = torch.zeros(n_nodes, dtype=torch.bool)
                train_mask[labeled_indexes] = True
               

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

                for epoch in range(1, self.epoch+1):

                    #-- log --
                    if epoch==1 or epoch%10 ==0 or epoch==self.epoch:
                        print('\tSampling Nieghbors ...')
                    
                    self.sampled_edges = []
                    self.sampled_weights = []

                    self.Samplig(self.k)


                    unique_edges = {}
                    for i, edge in enumerate(self.sampled_edges):
                        if tuple(edge) not in unique_edges:
                            unique_edges[tuple(edge)] = self.sampled_weights[i]



                    sampled_edge_list = list(unique_edges.keys())                    

                    sampled_edge_index = torch.tensor(sampled_edge_list, dtype=torch.long)
                    sampled_edge_index = sampled_edge_index.t().contiguous()                 

                    edge_index = sampled_edge_index        

                    #loss_train = train()
                    loss_train, embeddings = gat_util.train(model= gat2,
                                               optimizer= optimizer,
                                               device= self.device,
                                               x= self.X,
                                               edge_index= edge_index,
                                               y_train = y_train,
                                               train_mask= train_mask)
                    
                    
                    #acc_test , loss_test , macro_f1 = evaluate()
                    acc_test , loss_test, macro_f1 , predictions = gat_util.test(model= gat2,
                                                              device= self.device,
                                                              x= self.X,
                                                              edge_index= edge_index,
                                                              y_train= y_train,
                                                              y_true= self.true_labels,
                                                              labeled_indexes= self.labeled_indexes)
                    
                    
                    if epoch==1 or epoch%10 ==0 or epoch==self.epoch:
                        print(f'Epoch {epoch:03d}, Loss: {loss_train:.4f}, Unlabeled Acc: {acc_test:.4f}, Macro-f1: {macro_f1:.4f}')

                    train_losses.append(loss_train)
                    test_accs.append(acc_test)
                    test_losses.append(loss_test)

                    if macro_f1> best_macro:
                        best_macro = macro_f1
                        best_epoch = epoch
                        best_model = copy.deepcopy(gat2)
                
                
                #-- log --        
                print('\tFinish Train: Best Macro-F1: %f , Best Epoch: %d'
                          %(best_macro , best_epoch))     
                print('\tSaving Results ...')       
                
                torch.save(best_model.state_dict(), self.model_file)

                gat_util.fina_evaluate(iteration = self.iteration,
                                       k = self.k,
                                       alpha = self.alpha,
                                       aa_threshold = t,
                                       model = best_model,
                                       device = self.device,
                                       x = self.X,
                                       edge_index = edge_index ,
                                       Y_true = self.true_labels,
                                       labeled_indexes = self.labeled_indexes,
                                       results_file = self.results_file,
                                       method_name = 'LOSS-GAT',
                                       ds_name = self.ds_title,
                                       labeled_percent = self.labeled_percent,
                                       lp_percent = p)
    #--------------------------------------------------------------------------





        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
                            
                            
                            
        

