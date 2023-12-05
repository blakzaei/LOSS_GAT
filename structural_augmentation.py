#-- IMPORTS -------------------------------------------------------------------
from util import util

import numpy as np

import networkx as nx

from parmap import starmap
import os
###############################################################################

class Adamic_Adar_Augmentation:
    
    def __init__(self , k , alpha, labeled_percent, iteration,
                 ds_title, aa_thresholds = [float(i) / 10 for i in range(1, 7)],
                 number_of_processes = 5,
                 sub_pairs_length = 500): 
        
        #-- log --
        print('Start Structual Augmentation ---------------------------------')
        print('\titeration=%d - k=%d - alpha = %f' \
                  %(iteration,k,alpha))
        
        #-- initlize params ---------------------------------------------------
        self.k = k
        self.alpha = alpha           
        self.labeled_percent = labeled_percent
        self.iteration = iteration
        self.ds_title = ds_title     
        self.aa_thresholds = aa_thresholds
        
        
        self.input_path = 'results/' 
        
        self.indexes_file = self.input_path + 'doc2vec/indexes.npy'
        self.doc2vec_file = self.input_path + 'doc2vec/doc2vec_vect.csv'
        self.true_labels_file = self.input_path + 'doc2vec/labels.npy'
        
        self.graph_file = self.input_path + 'graph/knn_graph_' + str(self.k) + '.graphml'   
        
        self.output_path = 'results/sa/'
        util.create_folder(self.output_path)       
                
        
        self.number_of_processes = number_of_processes
        self.sub_pairs_length = sub_pairs_length        
              
                
        self.G = nx.read_graphml(self.graph_file)
        
        #-- Run ---------------------------------------------------------------
        self.run_aa_augmentation()        
        #self.create_edge_index()     
        
        #-- log --
        print('Finish Structual Augmentation --------------------------------')  
        
        
    #--------------------------------------------------------------------------
    def Compute_AA_Index(self, u, v):        
        common_neighbors = list(nx.common_neighbors(self.G, u, v))
        aa_index = sum(1 / np.log(self.G.degree(w)) for w in common_neighbors)
        return (u, v, aa_index)
    #--------------------------------------------------------------------------
    
    
    #--------------------------------------------------------------------------
    def run_aa_augmentation(self):
        
        #-- log --
        print('\tRunnig Augmenation ... ')
        
        nodes = self.G.nodes()
        
        print('\tNodes: ' , self.G.number_of_nodes())
        print('\tEdges: ' , self.G.number_of_edges())
    
        print('\tSetting Weight 1 for all edges:')
        nx.set_edge_attributes(self.G, 1, 'weight')
    
        print('\tCreating All Pairs of Nodes ...')        
        pairs = [(u, v) for u in nodes for v in nodes if not self.G.has_edge(u, v) and u!=v]
        print('\t-paris length = %d' %len(pairs))
    
        print('\tCalculating AA for All Pairs ..')
        counter = 1
        sub_pairs_length = self.sub_pairs_length
        max_aa_value = -1
    
        for i in range(0, len(pairs), sub_pairs_length):
            if i%sub_pairs_length == 0:
                print('\t\t-subset number: ' , i , '----')
    
            sub_pairs = pairs[i:i+sub_pairs_length]
            sub_results = starmap(self.Compute_AA_Index, sub_pairs,
                                  pm_processes=self.number_of_processes)
    
            max_aa = max(aa_val for u, v, aa_val in sub_results)
            if max_aa > max_aa_value:
                max_aa_value = max_aa
    
    
            #-- remove edges with aa<max to decrease file size --
            sub_results = [(u, v , float(aa_val)) for u, v, aa_val in sub_results \
                           if float(aa_val)/max_aa_value>=min( self.aa_thresholds)]
            print('\t\t-subset %d size after elimination: %d' %(counter,len(sub_results)))
    
            temp_result_file =  self.output_path + 'temp_' + str(counter)
            np.save(temp_result_file,sub_results)
            counter +=1        
            
    
    
    
        print('\tAdding New Edges By Threshold ...')
        for t in  self.aa_thresholds:
            G_copy =  self.G.copy()
            print('\n\t\tt=%f -----------------------------------' %(t))
    
            for i in range(1,counter):
                #print('\t-Loading temp file %d' %i)
                temp_result_file =  self.output_path + 'temp_' + str(i) + '.npy'
                sub_results = np.load(temp_result_file)
                sub_results = sub_results.tolist()  
               
                for u, v, aa_val in sub_results:
                    if float(aa_val)/max_aa_value <= t:
                        continue
    
                    if not  self.G.has_edge(u, v):
                        G_copy.add_edge(u, v, weight=aa_val)
                    else:
                        G_copy.add_edge(u, v, weight=1+aa_val, overwrite=True)   
            
    
            print('\tNodes After AA: ' ,G_copy.number_of_nodes())
            print('\tEdges After AA: ' , G_copy.number_of_edges())
            print('\tHow Many Edges were added?: ' ,(G_copy.number_of_edges()-self.G.number_of_edges()))
    
            
            print('\tSaving Augmented Graph ...')
            graphml_output_file = 'aa_knn_graph_' + str(self.k) + '_' + str(t) + '.graphml'
            nx.write_graphml(G_copy,  self.output_path + graphml_output_file )    
            
    
        print('\tRemoving Temp Files ...')
        for i in range(1,counter):
            temp_result_file =  self.output_path + 'temp_' + str(i) + '.npy'
            os.remove(temp_result_file)
            print('temp_' + str(i) + '.npy' + ' Removed. :)')
    #--------------------------------------------------------------------------   
    
    '''
    #--------------------------------------------------------------------------
    def create_edge_index(self):
        
        #-- log --
        print('\tCreating Augmented Edge Indexes ... ')
        
        for t in self.aa_thresholds:
              print('k=%d , t=%f =====================================================' %(self.k,t))
        
              new_graph_file = 'aa_knn_graph_' + str(self.k) + '_' + str(t) + '.graphml'
        
        
              G = nx.read_graphml(self.output_path + new_graph_file)
        
              print('Nodes: ' ,G.number_of_nodes())
              print('Edges: ' , G.number_of_edges())
        
              esdges = list(G.edges(data=True))
              
              df = pd.read_csv(self.doc2vec_file, index_col=0)
              print('df: ' , df.shape)
              print('Columns:\n' , df.columns)
              
              indexes = df.index.values.tolist()
        
              edge_list = []
              weight_list = []
        
              for e in esdges:
                  n1 = e[0]
                  n2 = e[1]
                  w = e[2]['weight']
        
        
                  index_1 = indexes.index(n1)
                  index_2 = indexes.index(n2)
        
                  e1 = [index_1,index_2]
                  e2 = [index_2,index_1]
        
                  if e1 not in edge_list:
                        edge_list.append(e1)
                        weight_list.append(float(w))
        
                  if e2 not in edge_list:
                        edge_list.append(e2)
                        weight_list.append(float(w))
        
        
        
        
              print('All Edges: ' , len(edge_list))
        
              #print(weight_list)
              edge_index = torch.tensor(edge_list, dtype=torch.long)
              weight_index = torch.tensor(weight_list, dtype=torch.float32)
        
              torch.save(edge_index, self.output_path + 'aa_edge_index_' + \
                         str(self.k) + '_' + str(t) + '.pt')
              torch.save(weight_index, self.output_path + 'aa_weight_index_' \
                         + str(self.k) + '_' + str(t) + '.pt')
        
              print('Edge List: Done ;)')
    #--------------------------------------------------------------------------
    '''




        













    

    



