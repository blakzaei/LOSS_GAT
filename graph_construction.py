#-- IMPORTS -------------------------------------------------------------------
from util import util

import pandas as pd
import numpy as np
import networkx as nx

from scipy.spatial.distance import cdist
from scipy.spatial.distance import cosine
###############################################################################

#-- Graph Class ---------------------------------------------------------------
class Graph():
    def __init__(self, k_values=[5,6,7] , alpha_values =[0.005,0.01,0.02]):

        #-- log --
        print('Start Graph Construction -------------------------------------')
        
        #-- initlize params ---------------------------------------------------
        self.k_values = k_values
        self.alpha_values = alpha_values    
        
        
        self.input_path = 'results/doc2vec/'        
        self.ids_file = self.input_path + 'ids.npy'        
        self.doc2vec_file = self.input_path + 'doc2vec_vect.csv' 
        
        
        self.output_path = 'results/graph/'
        util.create_folder(self.output_path)

        self.sim_matrix_file = self.output_path + 'similarity_matrix.csv'
        self.graphml_output_file = self.output_path + 'knn_graph_'        
        self.output_matrix_w = self.output_path +'w_matrix_'        

            
        self.min_weigth = float(0.08)  
        
        self.ids = np.load(self.ids_file)                
        self.doc2vec_df = pd.read_csv(self.doc2vec_file, sep=',', index_col=0, header=0)
        
        
        #-- Create Similarity Matrix ------------------------------------------
        self.similarity_matrix = self.calc_similarity_matrix()

        
        #-- Create KNN Graph and W(katz) Matrix -------------------------------
        self.A = None
        self.G = None
        
        for k in self.k_values:            
            self.A, self.G = self.create_knn_graph(k) 
            
            for alpha in self.alpha_values:
                self.create_w_katz_matrix(k, alpha)     
        
        
        #-- log --
        print('Finish Graph Construction -------------------------------------')


    #--------------------------------------------------------------------------    
    def calc_similarity_matrix(self):      
        
        #-- log --
        print('\tCalculating Similarity Matrix ...')

        sim_matrix = []
        for id in self.ids:
            sim_matrix.append(self.doc2vec_df.loc[id].tolist())

        sim_matrix = pd.DataFrame(sim_matrix, index=self.ids)    
        
        Y = cdist(sim_matrix, sim_matrix, metric=cosine)
        similarity_matrix = pd.DataFrame(Y, columns=self.ids, index=self.ids)
        
        #-- log --
        print('\t\tSimilarity Matrix:' , similarity_matrix.shape)
        print('\tSaving Similarity Matrix ...')
        
        similarity_matrix.to_csv(path_or_buf=self.sim_matrix_file)        
        
        return similarity_matrix
    #--------------------------------------------------------------------------   
    
    #--------------------------------------------------------------------------
    def create_knn_graph(self, k):

        #-- log --
        print('\tCreating KNN Graph for k=%d ...' %k)
            
        A = pd.DataFrame(0, columns=self.similarity_matrix.index, index=self.similarity_matrix.index)
        
        G = nx.Graph()
        for index_i, row in self.similarity_matrix.iterrows():			
            knn = [1000 for temp in range(k)]
            knn_names = ['' for temp in range(k)]
            max_value = 1000
            max_value_id = 0           
            
            for name_j, value in row.items():				
                if(index_i != name_j):					
                    if (value < max_value):						
                        knn_names[max_value_id] = name_j
                        knn[max_value_id] = value
                        max_value_id = np.argmax(knn)
                        max_value = knn[max_value_id]
           

            for j in range(k):				
                index_j = knn_names[j]
                A.loc[index_i][index_j] = 1
				
                G.add_edge(index_i, index_j, weight=1)

        #-- log --
        print('\t\tKNN Graph for k=%d: Number of Nodes=%d - Number of Edges=%d' 
              %(k , G.number_of_nodes(), G.number_of_edges()))
        print('\tSaving KNN Graph for k=%d ...' %k)
        
        nx.write_graphml(G, self.graphml_output_file + str(k) + '.graphml')        

        return A, G
    #--------------------------------------------------------------------------      
    
    #--------------------------------------------------------------------------
    def create_w_katz_matrix(self, k, alpha):
        
        #-- log --
        print('\tCreating W (katz) for k=%d and alpha=%f ...' %(k,alpha))
        
        matrix_w_output_name = self.output_matrix_w + str(k) + '_' + str(alpha) + '.csv'

        ident_matrix = self.calc_Indent()
        A = self.A.mul(alpha)
        
        I = ident_matrix.subtract(A)

        I = pd.DataFrame(np.linalg.pinv(I.values.astype(np.float32)), columns=self.ids, index=self.ids)
        W = I.subtract(ident_matrix)

        #-- log --
        print('\t\tW(katz):' , W.shape)
        print('\tSaving W (katz) for k=%d and alpha=%f ...' %(k,alpha))
        
        W.to_csv(path_or_buf=matrix_w_output_name)        
        del W
    #--------------------------------------------------------------------------
    
    #--------------------------------------------------------------------------    
    def calc_Indent(self):
        
        ident_matrix = np.identity(len(self.similarity_matrix), dtype = float)
        ident_matrix = pd.DataFrame(ident_matrix, columns=self.ids, index=self.ids)        

        return ident_matrix
    #--------------------------------------------------------------------------








