#-- IMPORT --------------------------------------------------------------------
import pandas as pd
import numpy as np

import torch
import torch.nn.functional as F

import networkx as nx

from sklearn.metrics import f1_score, accuracy_score, \
    confusion_matrix, precision_score, recall_score
###############################################################################


#-- load Features Vectors as X ------------------------------------------------
def create_X(doc2vec_file):
        
    #-- log --
    print('\tConverting Doc2Vec vectors to tensors as X ...')
    
    df_doc2vec = pd.read_csv(doc2vec_file, index_col=0)       
    
    X = df_doc2vec.iloc[:,:]
    X = np.array(X)
    X = torch.tensor(X, dtype=torch.float)
    
    #-- log --
    print('\tX:' , X.shape) 
    
    return X
#------------------------------------------------------------------------------

#-- Load True Labels ----------------------------------------------------------
def load_true_labels(true_labels_file):
    
    #-- log --
    print('\tLoading True Labels ...')
    
    Y_true = np.load(true_labels_file)
    Y_true = Y_true.astype(np.float32)
    Y_true = torch.tensor(Y_true, dtype=torch.float)
    Y_true= torch.reshape(Y_true, (Y_true.shape[0],1))

    print('\tY_true:' , Y_true.shape )
    
    return Y_true
#------------------------------------------------------------------------------

#-- Create Semi-Supervised Y for Training -------------------------------------
def create_y_semi_supervised(true_labels_file, rp , rn):
        
        true_labels = load_true_labels(true_labels_file)
        
        #-- set 1:fake , 0:real , -1:un-labeled
        y_train = np.zeros((true_labels.shape[0]))
        labeled_indexes = []        
        for i in range(y_train.shape[0]):
            if i in rp:
                y_train[i] = 1
                labeled_indexes.append(i)
                
            elif i in rn:
                y_train[i] = 0
                labeled_indexes.append(i)
                
            else:
                y_train[i] = -1

        y_train = torch.tensor(y_train)
        
        return y_train , labeled_indexes
#------------------------------------------------------------------------------

#-- Create Edge indexes from Graph File ---------------------------------------
def create_edge_indexes(graph_file, doc2vec_file, weighted = False):

        #-- log --
        print('\tCreating Edge Index for GAT ...')
        
        G = nx.read_graphml(graph_file)
    
        print('\tLoading KNN Graph ...')
        print('\t\tNodes: ' ,G.number_of_nodes())
        #print('\t\tEdges: ' , G.number_of_edges())
    
        esdges = list(G.edges(data=weighted))       
       
        
        df_doc2vec = pd.read_csv(doc2vec_file, index_col=0)
        indexes = df_doc2vec.index.values.tolist()
    
        edge_list = []
        weight_list = []
    
        for e in esdges:            
            n1 = e[0]
            n2 = e[1]
            
            if weighted:
                w = e[2]['weight']
    
            index_1 = indexes.index(n1)
            index_2 = indexes.index(n2)
    
            e1 = [index_1,index_2]
            e2 = [index_2,index_1]
    
            if e1 not in edge_list:
                edge_list.append(e1)
                
                if weighted:
                    weight_list.append(float(w))
    
            if e2 not in edge_list:
                edge_list.append(e2)
                
                if weighted:
                    weight_list.append(float(w))
    
    
        print('\t\tAll Edges: ' , len(edge_list))
    
        edge_index = torch.tensor(edge_list, dtype=torch.long)
        weight_index = torch.tensor(weight_list, dtype=torch.float32)
        
        #print('\tSaving Edge list ...')
        #torch.save(edge_index, self.output_path + 'edge_index_' + str(self.k) + '.pt')
        
        return edge_index , weight_index       
#------------------------------------------------------------------------------




#-- Define Train Function  ----------------------------------------------------
def train(model , optimizer , device , x , edge_index , y_train , train_mask):
    
    model.train()    
    optimizer.zero_grad() 
    
    logits, embeddings = model(x.to(device), edge_index.to(device))
    logits = logits.flatten()
    
    loss = F.binary_cross_entropy_with_logits(logits[train_mask],
                                              (y_train[train_mask] > 0).float())    
    loss.backward()
    optimizer.step()
    
    return loss.item() , embeddings
###############################################################################


#-- Define Test Function  -----------------------------------------------------
def test(model, device , x , edge_index , y_train, y_true , labeled_indexes):
    
    model.eval()
    
    with torch.no_grad():
        logits, embeddings = model(x.to(device), edge_index.to(device))
        logits = logits.flatten()
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float().squeeze()
        loss = F.binary_cross_entropy_with_logits(logits,
                                                  (y_train > 0).float())

        real_labels = y_true.cpu().squeeze()
        pred_results = preds.cpu()

        real_labels = torch.tensor([v for i, v in enumerate(real_labels)
                                    if i not in labeled_indexes])
        pred_results = torch.tensor([v for i, v in enumerate(pred_results)
                                     if i not in labeled_indexes])
        
        

        macro_f1 = f1_score(real_labels, pred_results, average='macro')

        acc = accuracy_score(real_labels, pred_results)

    return acc, loss , macro_f1 , preds
###############################################################################


#-- Define Final Test Function (When Training is finished) --------------------
def fina_evaluate(iteration, k , alpha , aa_threshold ,
                  model , device, x , edge_index , Y_true, labeled_indexes,
                  results_file, method_name, ds_name , labeled_percent , 
                  lp_percent):
    
    model.eval()
    with torch.no_grad():
        logits, embeddings = model(x.to(device), edge_index.to(device))
        logits = logits.flatten()

        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float().squeeze()

        real_labels = Y_true.cpu().squeeze()
        pred_results = preds.cpu()

        real_labels = torch.tensor([v for i, v in enumerate(real_labels)
                                    if i not in labeled_indexes])
        pred_results = torch.tensor([v for i, v in enumerate(pred_results)
                                     if i not in labeled_indexes])

        cm_val = confusion_matrix(real_labels, pred_results)
        macro_f1_val = f1_score(real_labels, pred_results, average='macro')
        micro_f1_val = f1_score(real_labels, pred_results, average='micro')        
        acc_val = accuracy_score(real_labels, pred_results)
        macro_pr_val = precision_score(real_labels, pred_results, average='macro')
        micro_pr_val = precision_score(real_labels, pred_results, average='micro')
        macro_re_val = recall_score(real_labels, pred_results, average='macro')
        micro_re_val = recall_score(real_labels, pred_results, average='micro')

        #-- Get F1 for Interset Class = fake samples
        tp_fake = cm_val[1, 1]  # True Positives for fakes
        fp_fake = cm_val[0, 1]  # False Positives for fakes
        fn_fake = cm_val[1, 0]  # False Negatives for fakes

        # Calculate precision and recall for Class 1
        precision_fake = tp_fake / (tp_fake + fp_fake)
        recall_fake = tp_fake / (tp_fake + fn_fake)

        # Calculate the F1 score for Class 1
        interest_f1_val = 2 * (precision_fake * recall_fake) / (precision_fake + recall_fake)

        #-- Save reults to cvs full results ------------------------------------

        df_results = pd.read_csv(results_file)
        results = {'DS_name' : ds_name,
                  'labeled_amount' : labeled_percent,
                  'iteration' : iteration,
                  'k' : k,
                  'alpha' : alpha ,
                  'aa_threshold' : aa_threshold ,                  
                  'lp2_selected_percent' : lp_percent,                  
                  'macro_f1' : macro_f1_val,
                  'interest_f1' : interest_f1_val,
                  'micro-f1' : micro_f1_val,                  
                  'macro-precision' : macro_pr_val,
                  'micro-precision' : micro_pr_val,
                  'macro-recall' : macro_re_val,
                  'micro-recall' : micro_re_val ,
                  'acc' : acc_val}

        new_df = pd.DataFrame(results, index=[0])
        df_results = pd.concat([df_results, new_df], ignore_index=True)   
        df_results.to_csv(results_file, index=False)
        
        
    
###############################################################################
