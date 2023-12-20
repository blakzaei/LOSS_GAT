#-- IMPORT -------------------------------------------------------------------
###############################################################################
import sys

import feature_extraction
import split_train_test
import graph_construction
import label_propagation_katz as lp_katz
import label_propagation_initial_classifier as lp_cls
import structural_augmentation as sa
import classification
###############################################################################

'''
DS_TITLE : DS_LANGUAGE

fnn: english

fakeBr: portuguese
fact_checked_news: portuguese

fake_news_data: english
fake_news_detection: english

'''


#AMOUNT_LABELD = 0.1
#N_ITERATIONS = 10

K_VALUES = [5,6,7]
ALPHA_VALUES =[0.005,0.01,0.02]

#EPOCHS = 100
###############################################################################

def main(ds_title , amount_labeled, n_iterations, epochs):
    
    ds_languge = 'english'    
    if ds_title.lower()== 'fakebr' or ds_title.lower()== 'fact_checked_news':
        ds_languge = 'portuguese'        

    feature_extraction.Doc_2_Vec(ds_title = ds_title,
                                 ds_languge = ds_languge)
    
    graph_construction.Graph(K_VALUES , ALPHA_VALUES)
    
    split_train_test.split(ds_title = ds_title,
                           labeled_percent = amount_labeled,
                           number_of_iterations=n_iterations)
    
    for k in K_VALUES:
            for alpha in ALPHA_VALUES: 
                sa.Adamic_Adar_Augmentation(k=k,
                                            alpha=alpha,
                                            labeled_percent=amount_labeled,                                            
                                            ds_title= ds_title)
                
    
    
    for iteration in range(1, n_iterations+1):
        for k in K_VALUES:
            for alpha in ALPHA_VALUES:           
                lp_katz.LP_Katz(k=k,
                                alpha=alpha,
                                labeled_percent=amount_labeled,
                                iteration=iteration,
                                ds_title= ds_title)
    
                lp_cls.LP_Classifier(k=k,
                                     alpha=alpha,
                                     labeled_percent=amount_labeled,
                                     iteration=iteration,
                                     ds_title= ds_title,
                                     epoch = epochs)              
                
                classification.Classification(k=k,
                                              alpha=alpha,
                                              labeled_percent=amount_labeled,
                                              iteration=iteration,
                                              ds_title= ds_title,
                                              epoch = epochs)

###############################################################################

if __name__ == "__main__":
    
    if len(sys.argv) != 5:
        print('Usage: python main.py <ds_title> , <amount_labeled>, <n_iterations>, <epochs>')
        sys.exit(1)
    
    ds_title = sys.argv[1]
    amount_labeled = float(sys.argv[2])
    n_iterations = int(sys.argv[3])
    epochs = int(sys.argv[4])
    
    msg = f'''
            Start LOSS_GAT on Dataset: {ds_title}\n
            {amount_labeled} amount of fake samples are labeled\n
            number of epochs for tarining networks is {epochs}\n
            after {n_iterations} iteration, results will be evaluated.
        '''
    
    print(msg)
    
    
    main(ds_title , amount_labeled, n_iterations, epochs)




    



