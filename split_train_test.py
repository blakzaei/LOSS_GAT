#-- IMPORT --------------------------------------------------------------------
from util import util

import numpy as np
import pandas as pd
import random
###############################################################################

#-- Set Positive and Unlabed data ---------------------------------------------
def get_train_teste(dataset, percent):      

        fakes = dataset[dataset['label']==1]
        fake_indexes = fakes.index

        n_fakes = fakes.shape[0]
        n_labeled = int(percent * n_fakes)
        random_labeled = random.sample(range(0, n_fakes), n_labeled)

        print('\t\tNumber of fakes:' , n_fakes)
        print('\t\tNumber of labeled:' , n_labeled)

        labeled_indexes = []
        for r in random_labeled:
            labeled_indexes.append(fake_indexes[r])          

        
        return labeled_indexes
    


#-- Get Positive ans Unlabeled Data --------------------------------------------
###############################################################################
def split(ds_title, labeled_percent, number_of_iterations):
    
    #-- log --
    print('Start Splitting Data to Labeled and Unlabeled --------------------')
    
    ds_path = 'datasets/' + ds_title + '/'
    ds_file = ds_path + ds_title + '.csv'   

    output_path = 'results/split/'
    util.create_folder(output_path)    
    
    df = pd.read_csv(ds_file)    
    print('\tdataset size:' , df.shape)   
    
    iterations = list(range(1,number_of_iterations+1))
    
    for iter in iterations:
        print('\tIteration: %d ........................................' %iter)
    
        labeled_indexes = get_train_teste(df, labeled_percent)
    
        #-- Save labeled and unlabeld  ----------------------------------------
        labeled_indexes_file = output_path + str(iter) + '_' + str(labeled_percent) + \
            '_Labeled_indexes'    
        np.save(labeled_indexes_file , labeled_indexes)       
    
        #-- Save Labeled Ids --------------------------------------------------  
        labeled_df = df.loc[labeled_indexes]        
        print('\t\tlabeled_df:' , labeled_df.shape)        
    
        labeled_ids = labeled_df['id'].tolist()         
        labeled_ids_file = output_path +  str(iter) + '_' + str(labeled_percent) + '_labeled_ids'
        np.save(labeled_ids_file , labeled_ids)       
        
        #-- log --
        print('Finish Splitting Data to Labeled and Unlabeled --------------------')
        
        
        
        