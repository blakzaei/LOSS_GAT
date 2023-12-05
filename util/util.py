#-- IMPORT --------------------------------------------------------------------
import os

import pandas as pd
import numpy as np

import torch
###############################################################################

#-- Create Folders ------------------------------------------------------------
def create_folder(name=""):
    parent_folder_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    new_directory_path = os.path.join(parent_folder_path, name)    
    
    if not os.path.exists(new_directory_path):
        os.makedirs(new_directory_path)  
#------------------------------------------------------------------------------




#-- Create an Empty Dataframe for best results ----------------------------
def create_empty_df_for_results(results_file):        
    cols_names = ['DS_name',
                  'labeled_amount',
                  'iteration',
                  'k' ,
                  'alpha' ,
                  'aa_threshold' ,                  
                  'lp2_selected_percent',                  
                  'macro_f1',                  
                  'micro-f1',                  
                  'macro-precision' ,
                  'micro-precision',
                  'macro-recall' ,
                  'micro-recall' ,
                  'acc']
    
    df_results = pd.DataFrame(columns=cols_names)
    df_results.to_csv(results_file, index=False)
#--------------------------------------------------------------------------



