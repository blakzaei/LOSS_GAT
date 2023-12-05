#-- IMPORTS -------------------------------------------------------------------
###############################################################################
from util import util

import numpy as np
import pandas as pd
import networkx as nx


#-- LP_Katz Class -------------------------------------------------------------
class LP_Katz():
    
    def __init__(self , k , alpha, labeled_percent, iteration, ds_title):

        #-- log --
        print('Start Label Propagation Using Katz Index ---------------------')
        print('\titeration=%d - k=%d - alpha = %f' \
                  %(iteration,k,alpha))
        
        #-- initlize params ---------------------------------------------------
        self.k = k
        self.alpha = alpha           
        self.labeled_percent = labeled_percent
        self.iteration = iteration
        self.ds_title = ds_title 
        
        
        self.input_path = 'results/'   
        
        self.ds_file = 'datasets/' + self.ds_title + '/' + self.ds_title + '.csv'  
        self.graphml_input_file = self.input_path + 'graph/knn_graph_' + \
            str(self.k) + '.graphml'
        self.matrix_w_input_file =  self.input_path +'graph/w_matrix_' + \
            str(self.k) + '_' + str(self.alpha) + '.csv'        
        self.labeled_ids_file = self.input_path + 'split/' +  str(self.iteration) \
            + '_' + str(self.labeled_percent) + '_labeled_ids.npy'         
              
        
        self.output_path = 'results/lp_katz/'
        util.create_folder(self.output_path)

        self.output_file_rp = self.output_path + str(self.iteration) + '_' + \
            str(self.labeled_percent) + '_rp_katz_' +  str(self.k) + '_' + \
                str(self.alpha) + '.text'
        self.output_file_rn = self.output_path + str(self.iteration) + '_' + \
            str(self.labeled_percent) + '_rn_katz_' +  str(self.k) + '_' + \
                str(self.alpha) + '.text'
        
        
        self.m = 1
        self.lmbda = 0.6        

        self.G = nx.read_graphml(self.graphml_input_file)
        self.W = pd.read_csv(self.matrix_w_input_file, index_col=0)      
        
        #-- run ---------------------------------------------------------------
        self.begin()

        
        #-- log --
        print('Finish Label Propagation Using Katz Index ---------------------')

    #--------------------------------------------------------------------------
    
    #--------------------------------------------------------------------------
    def begin(self):

        #-- log --
        print('\tLoading Dataset %s ...' %self.ds_title)        
        
        df = pd.read_csv(self.ds_file)
        print('\tDataset Size:' , df.shape)   
        
        labeled_ids = np.load(self.labeled_ids_file).tolist()        
        
        self.labeled_df = df[df['id'].isin(labeled_ids)]
        self.unlabeled_df = df[~df['id'].isin(labeled_ids)]

        #-- log --
        print('\tLabeled Samples:' , self.labeled_df.shape[0])
        print('\tUn-Labeled Samples:' , self.unlabeled_df.shape[0])       

        self.calc_rp_rn()
    #--------------------------------------------------------------------------
    
    #--------------------------------------------------------------------------
    def calc_rp_rn(self):
        
        #-- Positive (Fake) Labeled Samples --        
        P = list(self.labeled_df['id'].values)        
        
        #-- Unlabeled Samples --        
        U = list(self.unlabeled_df['id'].values)       
		
        RP = [] 
        P_copy = P[:]
        U_copy = U[:]

        #-- log --
        print('\tCalculating Ranks ...')
        
        top = int((self.lmbda / self.m)*len(P))        
        for self.k in range(0, self.m):			
            rank = np.zeros(top)
            rank_names = ['' for temp in range(top)]
            min_value = 0
            min_value_id = 0
            for vi in U_copy:
                Svi = 0
                for vj in P_copy:          

                    Svi += self.W.loc[vi][vj]

                Svi /= len(P_copy)
                if (Svi > min_value):
                    rank_names[min_value_id] = vi
                    rank[min_value_id] = Svi
                    min_value_id = np.argmin(rank)
                    min_value = rank[min_value_id]
            
            for i in rank_names:
                P_copy.append(i)
                U_copy.remove(i)
                RP.append(i)
                U.remove(i)        
        
        bottom = len(RP+P)        
        rank = [10000 for temp in range(bottom)]
        RN = ['' for temp in range(bottom)]
        max_value = 10000
        max_value_id = 0

        for vi in U:
            Svi = 0
            for vj in (P + RP):
                Svi += self.W.loc[vi][vj]

            Svi /= len(P + RP)
            if (Svi < max_value):
                RN[max_value_id] = vi
                rank[max_value_id] = Svi
                max_value_id = np.argmax(rank)
                max_value = rank[max_value_id]       

        
        #-- log --
        print('\tSize of RP_katz: %d' %len(RP+P))
        print('\tSize of RN_katz: %d' %len(RN))
        print('\tSaving RP_katz and RN_katz ...')        
        
        self.save_file(RP+P, 'rp')
        self.save_file(RN, 'rn')
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    def save_file(self, data , t):

        if t=='rp':
            file_name = self.output_file_rp            
        
        else:
            file_name = self.output_file_rn
           

        f = open(file_name, 'w')
        for i in data:
            f.write(str(i) + '\n')

        f.close()
    #--------------------------------------------------------------------------
    









