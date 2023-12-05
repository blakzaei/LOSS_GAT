#-- IMPORT -------------------------------------------------------------------
###############################################################################
from util import util

import pandas as pd
import numpy as np

import nltk
from nltk import word_tokenize
from nltk import download
from nltk.corpus import stopwords
from nltk.stem.porter import *
nltk.download('punkt')
download('stopwords')
nltk.download('rslp')

import string

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
###############################################################################

#-- Doc2Vec Class -------------------------------------------------------------
class Doc_2_Vec():
    def __init__(self, ds_title, ds_languge='english'): 
        
        #-- log --
        print('Start Feature Extration by Doc2Vec ---------------------------')
        
        #-- initlize params ---------------------------------------------------
        self.ds_title = ds_title
        self.ds_languge = ds_languge
        
        
        self.ds_path = 'datasets/' + ds_title + '/'
        self.ds_file = self.ds_path + ds_title + '.csv'         
       
        self.stopwords_file = self.ds_path + 'stopwords.txt'        
        
        self.output_path = 'results/doc2vec/'
        util.create_folder(self.output_path)                    

        self.domain_stopwords = self.get_domain_stopwords(self.stopwords_file)
        
        self.documents = None
        self.ids = None
        self.labels = None
        self.tfidf_vect_df = None
        
        #-- run ---------------------------------------------------------------
        self.begin()
        
        #-- log --
        print('Finish Feature Extration by Doc2Vec --------------------------')
    #-------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    def begin(self):        

        self.preprocess()       
        
        #-- Create Model ------------------------------------------------------
        #-- log --
        print('\tCreating Doc2Vec Model ...')
        
        data = self.documents
        tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()),
                              tags=[str(i)]) for i, _d in enumerate(data)]

        dim_size=500
        window_size=8
        num_threads=4
        min_count=1
        alpha=0.025
        min_alpha=0.0001
        
        model = Doc2Vec(vector_size = dim_size,
                        alpha= alpha,
                        min_alpha= min_alpha,
                        min_count= min_count,
                        window= window_size,
                        workers= num_threads,
                        dm=1)

        model.build_vocab(tagged_data)
        
        model.train(tagged_data,
                    total_examples=model.corpus_count,
                    epochs=model.epochs)

        
        #-- log --
        print('\tSaving Doc2Vec Model ...')
        
        model.save(self.output_path + 'd2v.model')       

        #-- Extract Feature Vectors -------------------------------------------
        #-- log --
        print('\tExtracting Feature Vectors ...')
        
        number_of_data = len(self.documents)        
        number_of_features = dim_size

        X = np.zeros((number_of_data, number_of_features))

        for i in range(len(data)):
            X[i] = model.dv[str(i)]        
        
        X = pd.DataFrame(X)
        ids = pd.DataFrame(self.ids)        
        #Y = pd.DataFrame(self.labels)

        df_result = pd.concat([ids, X] , axis=1)        

        #-- log --
        print('\tSaving Feature Vectors as df: ' , df_result.shape , '...')
        df_result.to_csv(self.output_path + 'doc2vec_vect.csv', sep=',', encoding='utf-8', index=False)
    #-------------------------------------------------------------------------

    #-------------------------------------------------------------------------
    def preprocess(self):                
        
        #-- log --
        print('\tPreprocessing news texts ...')
        
        df = pd.read_csv(self.ds_file)        
        
        self.ids = df['id'].tolist()        
        self.labels =  df['label'].tolist()        
        
        docs = df['txt'].tolist()
        docs_removed_stop_words = list(map(self.remove_stopwords, docs))
        docs_steemed = list(map(self.stemming, docs_removed_stop_words))
        self.documents = [s.strip() for s in docs_steemed]
        
        np.save(self.output_path +'/ids', self.ids)
        np.save(self.output_path +'/documents', self.documents)
        np.save(self.output_path +'/labels', self.labels)    
    #-------------------------------------------------------------------------

    #-------------------------------------------------------------------------
    def remove_stopwords(self,text):    

        stop_words = nltk.corpus.stopwords.words(self.ds_languge)
        s = str(text).lower() 
        table = str.maketrans({key: None for key in string.punctuation})
        s = s.translate(table) 
        tokens = word_tokenize(s) 
        v = []
        for i in tokens:
            if not (i in stop_words or i in self.domain_stopwords or i.isdigit() or len(i)<= 1): # remove stopwords
                v.append(i)
        s = ""
        for token in v:
            s += token + " "   

        return s
    #-------------------------------------------------------------------------
    
    #-------------------------------------------------------------------------
    def stemming(self, text):        

        stemmer = PorterStemmer() # stemming for English
        if self.ds_languge =='portuguese':
            stemmer = nltk.stem.RSLPStemmer() # stemming for portuguese
            
        tokens = word_tokenize(text)
        sentence_stem = ''
        doc_text_stems = [stemmer.stem(i) for i in tokens]
        for stem in doc_text_stems:
            sentence_stem += stem+" "       

        return sentence_stem.strip()
    #-------------------------------------------------------------------------
    
    
    #-------------------------------------------------------------------------
    def convert(self , vector, type):
        vector = [type(i) for i in vector]
        return vector
    #-------------------------------------------------------------------------
    
    #-------------------------------------------------------------------------
    def get_domain_stopwords(self, stopwords=None):
        
        domain_stopwords = ''
        if stopwords==None:
            return domain_stopwords
        
        try:
            f = open(stopwords, 'r')
            for line in f:
                domain_stopwords +=  line.strip() + ' '
            f.close()
        
        except Exception as e:
            return ''
            
        return word_tokenize(domain_stopwords)
    #-------------------------------------------------------------------------

###############################################################################










