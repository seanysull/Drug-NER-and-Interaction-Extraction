# -*- coding: utf-8 -*-
"""
Created on Wed May  2 17:18:33 2018

@author: sean.o.sullivan
"""
#%%
import nltk
import xml.etree.ElementTree as ET
import pandas as pd
import os
import string
from nltk.tokenize import TreebankWordTokenizer
import numpy as np
from numpy.random import random_sample
import re
import pickle

from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers import Embedding, Dense, Dropout, Flatten, Merge
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import to_categorical

#%%

# =============================================================================
# tree2 = ET.parse('Data/Train/DrugBank/Alprazolam_ddi.xml')
# root2 = tree2.getroot()
# sents2 = list(root2)
# s1=sents2[0]
# ents = s1.findall("entity")
# pairs=s1.findall("pair")
# text=s1.attrib['text']
# =============================================================================

#%%
def replaceEntitiesDrugBlind(pair_tuple, entity_list,sentence):
    """ Function to replace non-target entities in sentence with neutral token to avoid confusion
        entity_dict contains all the entity non-targets to replace.   
    """
    replacement_term_dict = {pair_tuple[0]:"target_substance1",
                             pair_tuple[1]:"target_substance2"}
    
    if entity_list != None:
        for entity in entity_list:
            if entity not in replacement_term_dict:
                replacement_term_dict[entity] = "non_target_substance"
        
    entities = list(replacement_term_dict.keys())
    entities.sort(key=len, reverse=True)
   
    for ent in entities:
        replacement_term = replacement_term_dict[ent]
        sentence = sentence.replace(ent, replacement_term)
           
    return sentence
    
#%%
def tokeniseForDistance(sentence):
    """ Function to return tokens from a sentence       
    """
    punc = list(string.punctuation)
    tokens = TreebankWordTokenizer().tokenize(sentence)
    #tokens = [token for token in tokens if token not in punc]
       
    return tokens
   
#%%
def processTokenisedForWordEmbed(tokenised_sentence):
    """ Function to return tokens with preprocessing to fix mentions of sunstances
        in which we see things like drug1/drug2 or drug1 - induced, basically I tried to remove 
        combinations of words and punctuation that taken as a single token 
        would not appear in the embedding dictionary
    """
    tokens_final = list(tokenised_sentence)
    for token in tokenised_sentence:
        blind_terms = ['non_target_substance','target_substance1','target_substance2']
        break_flag = 0
                        
        # split terms that have a slash between them
        if re.match(r'\w*/\w*',token) != None:
           insertion_index = tokens_final.index(token) 
           split_tokens = split_token(token,'/')
           tokens_final = insert_tokens(tokens_final, insertion_index, split_tokens)
           continue
       
        if re.match(r'\w*~\w*',token) != None:
           insertion_index = tokens_final.index(token) 
           split_tokens = split_token(token,'~')
           tokens_final = insert_tokens(tokens_final, insertion_index, split_tokens)
           continue
 
   
        # split terms that have a full stop 
        if re.match(r'\w+\.\w+',token) != None:
           insertion_index = tokens_final.index(token) 
           split_tokens = split_token(token,'.')
           tokens_final = insert_tokens(tokens_final, insertion_index, split_tokens)
           continue
       

   
        for term in blind_terms:
            if term + "." in token:
               insertion_index = tokens_final.index(token) 
               split_tokens = split_token(token,'.')
               tokens_final = insert_tokens(tokens_final, insertion_index, split_tokens)
               break_flag = 1
               break
           
        if break_flag == 1: continue 
        # get rid of plurals
        for term in blind_terms:
            if term + "s" in token:
               token_to_insert = []
               token_to_insert.append(term) 
               insertion_index = tokens_final.index(token) 
               tokens_final = insert_tokens(tokens_final, insertion_index, token_to_insert)
               break_flag = 1
               break
           
        if break_flag == 1: continue
    
        # split hyphenated terms like drug1-inducing
        for term in blind_terms:
            if  term + '-' in token:
               insertion_index = tokens_final.index(token) 
               split_tokens = split_token(token,'-')
               tokens_final = insert_tokens(tokens_final, insertion_index, split_tokens)
               break_flag = 1
               break
            
        if break_flag == 1: continue   

        for term in blind_terms:
            if  term + '*' in token:
               insertion_index = tokens_final.index(token) 
               split_tokens = split_token(token,'*')
               tokens_final = insert_tokens(tokens_final, insertion_index, split_tokens)
               break_flag = 1
               break
            
        if break_flag == 1: continue 
    
        # split hyphenated terms like pre-drug1
        for term in blind_terms:   
            if '-' + term in token:
               insertion_index = tokens_final.index(token) 
               split_tokens = split_token(token,'-')
               tokens_final = insert_tokens(tokens_final, insertion_index, split_tokens)
               break
            
    
# =============================================================================
#     
#     TODO - psuedo"targetsubstance"
#     TODO - dihydroergotamine and ergotamine
#     
# =============================================================================
    tokens_final_lowercase = []
    for token in tokens_final:
        token_lower = token.lower()
        tokens_final_lowercase.append(token_lower)
    return tokens_final_lowercase
#%% 
def split_token(token,separator):
    """split a compound token on a given piece of punctuation eg a hyphen"""
    #pass separator as a string in quotes
    split = token.split(separator)
    if split[1] == None:
        insertion_tokens = [',',split[0]]   
    else:
        split.reverse()
        insertion_tokens = split
    return insertion_tokens   
#%%
def insert_tokens(tokenised_sentence, insertion_index, tokens_to_insert):
    # insert a set of split tokens into snetence for word embedding 
    insert_length = len(tokens_to_insert)
    
    if insert_length > 1:
        for tok in tokens_to_insert:
            tokenised_sentence.insert(insertion_index,tok)
    else:
        tokenised_sentence.insert(insertion_index,tokens_to_insert[0])
        
    del tokenised_sentence[insertion_index + insert_length]
    
    return tokenised_sentence
    
#%% 
def buildEntityDict(entity_list): 
    """function to build a list of enities with id as key and name and offset as val, 
       pass a list of entity elements """
    entity_name_dict = {}
    for entity in entity_list:
    # build dictionary of offset:entity type for all entities in sentence
        entity_name = entity.attrib.get('text')
        entity_id = entity.attrib.get('id')
        entity_name_dict[entity_id] = entity_name
        
    return entity_name_dict     
#%%

#%% 
def getRelativeDistances(tokenised_sentence):
    """Function to return two vectors of relative distances of words from drug1 and drug2
    with 0 padding up to sentence_max_length
    """
    # hard coded max length
    sentence_max_length = 155
    enumerated_tokens = list(enumerate(tokenised_sentence))
    for index, token in enumerated_tokens:
        if "target_substance1" in token: drug1_index = index
        if "target_substance2" in token: drug2_index = index
    dist_vector1 = [abs(token[0] - drug1_index) for token in enumerated_tokens]
    dist_vector2 = [abs(token[0] - drug2_index) for token in enumerated_tokens]
# =============================================================================
#     dist_vector1.remove(0)
#     dist_vector2.remove(0)
# =============================================================================
    sentence_length = len(dist_vector1)
    
    for i in range(sentence_max_length - sentence_length):
        dist_vector1.append(999)
        dist_vector2.append(999)
    
    dist_vector1 = np.array(dist_vector1).reshape(1,-1)
    dist_vector2 = np.array(dist_vector2).reshape(1,-1)

    d1d2 = np.concatenate([dist_vector1,dist_vector2],axis=1)
    
    return d1d2

#%%


def parseDDI(pathtoxml):
    """Fucntion to parse the xml documents expects to be passed 
      a path to the xml file to parse"""
    #parse tree using Elementtree
    tree = ET.parse(pathtoxml)
    #get the root of the tree
    root = tree.getroot()

    data_rows_per_pair = []
    
    sents = [elem for elem in tree.iter("sentence")]
    
    DocumentId = root.attrib.get("id")
    
    
    #import ipdb; ipdb.set_trace()
    for sentence in sents:
        # get sentence id and text
        sentId = sentence.attrib.get("id")
        senText = sentence.attrib.get("text")
        #get pairs and entities
        pairs = sentence.findall("pair")
        entities = sentence.findall("entity")
        
        # build dict of entity details to aid in preparing sentence text for specific entity pairs
                   
            
        if pairs == []:
            continue        
        # if ntities is not empty proceed to add tags for words as appropriate
        elif pairs != []:
                        
            for pair in pairs:
                # get dict of all entities in sentence
                entity_dict = buildEntityDict(entities)
                # get ids of entities in pair in question
                pair_entity1_id = pair.attrib.get("e1")
                pair_entity2_id = pair.attrib.get("e2")
                pair_entity1_name = entity_dict[pair_entity1_id]
                pair_entity2_name = entity_dict[pair_entity2_id]
                # skip apirs that are the same 
                if pair_entity1_name.lower() == pair_entity2_name.lower():
                    continue
                pair_id = pair.attrib.get("id")
                pair_interaction_type = pair.attrib.get("type")

                pair_name_tuple = (pair_entity1_name,pair_entity2_name)
                
                if pair_entity1_name not in senText or pair_entity2_name not in senText:
                    continue
                
                # remove them from the entity dict
                del entity_dict[pair_entity1_id]
                del entity_dict[pair_entity2_id]
                
                # get list of entity names
                replacement_targets = list(entity_dict.values())
                # call replacement function
                sentence_with_replacements = replaceEntitiesDrugBlind(pair_name_tuple, 
                                                                      replacement_targets,
                                                                      senText)
                IsInteraction = pair.attrib.get("ddi")
                
                if IsInteraction == "true": 
                    
                    IsInteraction = 1
                else: 
                    IsInteraction = 0
            
                
                tokenised_sentence = tokeniseForDistance(sentence_with_replacements)
                tokenised_sentence_w2v_1 = processTokenisedForWordEmbed(tokenised_sentence)
                tokenised_sentence_w2v = processTokenisedForWordEmbed(tokenised_sentence_w2v_1)

                shape_tokenised = len(tokenised_sentence_w2v)
                # check if token is awkward formulation of drug name:
                # plurals, drug1/drug2, drug1-induced etc....
                t1 = "target_substance1"
                t2 = "target_substance2"
                t3 = "non_target_substance"
                awk_flag = 0

                for token in tokenised_sentence_w2v:
                    if t1 in token and token != t1:
                        awk_flag = 1
                
                    elif t2 in token and token != t2:
                        awk_flag = 1
                        
                    elif t3 in token and token != t3:
                        awk_flag = 1
                        
                relative_dist_vec = getRelativeDistances(tokenised_sentence_w2v)
                
                rshape = relative_dist_vec.shape
                
                datarow = {"DocumentId":DocumentId,"SentenceId":sentId, 
                           "Sentence":senText,
                           "PairId": pair_id,
                           "Entity1id": pair_entity1_id,
                           "Entity2id": pair_entity2_id,
                           "Entity1name": pair_entity1_name,
                           "Entity2name": pair_entity2_name,                       
                           "IsInteraction":IsInteraction,
                           "InteractionType":pair_interaction_type,
                           "TokenisedEmbed": tokenised_sentence_w2v,
                           "Distances": relative_dist_vec,
                           "shapes":rshape,
                           "IsAwk": awk_flag,
                           "tokenShapes": shape_tokenised
                           }
            
                data_rows_per_pair.append(datarow)
                    
    df_doc = pd.DataFrame(data_rows_per_pair)
    
    return df_doc
#%%
def parseAndPickleData(data_path, pickled_path):
    """collect the full filenames of the drugbank/medline train/test where test is split into NER and DDI subtasks
    and build the dataframes, drugbank and medline are separate in case we wish to compare results against eachother.
    dataframes are preliminary ready to be processed to priduce final train/test sets.
    """
    # get list of files to process
    list_of_files = os.listdir(data_path)
    # create list of paths for files
    file_paths = [os.path.join(data_path,f) for f in list_of_files]
    # set up empyt list to put dataframes of parsed documents
    list_of_parsed_document_dataframes = []
    # iterate over paths and parse documents then add them to list
    for file in file_paths:
        parsed_doc_dataframe = parseDDI(file)
        #append entity df to entities and pairs to pairs
        list_of_parsed_document_dataframes.append(parsed_doc_dataframe)
    # concat dfs together and pickle file for       
    dataframe_of_all_parsed_documents = pd.concat(list_of_parsed_document_dataframes)
    dataframe_of_all_parsed_documents.to_pickle(pickled_path)

#%%    
def runParseAndPickle():
    # Drugbank train
    parseAndPickleData('C:\\Users\\sean.o.sullivan\\Documents\\uni\\ANLP\\Data\\Train\\DrugBank',
                       'C:\\Users\\sean.o.sullivan\\Documents\\uni\\ANLP\\Data\\Train\\DrugBank_DDI_Train.pkl')
    # Medline train
    parseAndPickleData('C:\\Users\\sean.o.sullivan\\Documents\\uni\\ANLP\\Data\\Train\\MedLine',
                       'C:\\Users\\sean.o.sullivan\\Documents\\uni\\ANLP\\Data\\Train\\MedLine_DDI_Train.pkl')
    # Drugbank test
    parseAndPickleData('C:\\Users\\sean.o.sullivan\\Documents\\uni\\ANLP\\Data\\Test\\TestDDI\\DrugBank',
                       'C:\\Users\\sean.o.sullivan\\Documents\\uni\\ANLP\\Data\\Test\\DrugBank_DDI_Test.pkl')
    # Medline test
    parseAndPickleData('C:\\Users\\sean.o.sullivan\\Documents\\uni\\ANLP\\Data\\Test\\TestDDI\\MedLine',
                       'C:\\Users\\sean.o.sullivan\\Documents\\uni\\ANLP\\Data\\Test\\MedLine_DDI_Test.pkl')

#%% 
def loadTrainTest():
    trainDrugBank = pd.read_pickle('C:\\Users\\sean.o.sullivan\\Documents\\uni\\ANLP\\Data\\Train\\DrugBank_DDI_Train.pkl')    
    testDrugBank = pd.read_pickle('C:\\Users\\sean.o.sullivan\\Documents\\uni\\ANLP\\Data\\Test\\DrugBank_DDI_Test.pkl')    
    trainMedline = pd.read_pickle('C:\\Users\\sean.o.sullivan\\Documents\\uni\\ANLP\\Data\\Train\\Medline_DDI_Train.pkl')    
    testMedline = pd.read_pickle('C:\\Users\\sean.o.sullivan\\Documents\\uni\\ANLP\\Data\\Test\\Medline_DDI_Test.pkl')    

    trainBoth = pd.concat((trainDrugBank,trainMedline))
    trainBoth.reset_index(drop=True,inplace=True)
    testBoth = pd.concat((testDrugBank,testMedline))
    testBoth.reset_index(drop=True,inplace=True)

    return trainBoth,testBoth        


#%% 
def buildPositionEmbedding(input_array, embedding_size):
    """Input Array is a tensor of the form (sentence,word)
    Output Array is a tensor of the form (sentence,word,embedding)
    
    """
    model = Sequential()
    model.add(Embedding(1000, embedding_size, input_length=310))
    model.compile('rmsprop', 'mse')
    output_array = model.predict(input_array)
    pad_length = int(output_array.shape[1]/2)
    dist_word_one = output_array[:,pad_length:,:]
    dist_word_two = output_array[:,:pad_length,:]
    dist_vecs = np.concatenate([dist_word_one,dist_word_two],axis=-1)
    return dist_vecs
#%%
def buildWordEmbeddingsDict():
    
    embeddings_dict = {}
    glove_data = 'C:/Users/sean.o.sullivan/Documents/uni/ANLP/glove.6B/glove.6B.50d.txt'
    glove_dimension = 50
    f = open(glove_data, encoding="utf-8")
    irregular_entries = []
    line_num = 0
    for line in f:
        values = line.split()
        word = values[0]
        try:
            float(first_value)
        except ValueError:
            irregular_entries.append(values)
            continue
        value = np.asarray(values[1:], dtype='float32')
        embeddings_dict[word] = value
    f.close()
    
    a = -2
    b = 2
    t1 = (b - a) * random_sample((50,)) + a
    t2 = (b - a) * random_sample((50,)) + a
    t3 = (b - a) * random_sample((50,)) + a
    
    blind_terms = {'target_substance1' : t1,
                   'target_substance2' : t2,
                   'non_target_substance' : t3,
                   'padding_term' : np.zeros((50,))                   
                  }
    
    embeddings_dict.update(blind_terms)
    
    print('Loaded %s word vectors.' % len(embeddings_dict))

    with open('word_embeddings_dict.pickle', 'wb') as handle:
        pickle.dump(embeddings_dict , handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    return embeddings_dict
#%%
def buildWordEmbeddingSentence(tokenised_sentence,embedding_dict):
    """given a tokenised sentence return a mtrix where aech row is the 
       appropriate word embedding"""
    
    #first pad the sentence with padding_term up to length 90
    max_sentence_length = 155
    current_sentence_length = len(tokenised_sentence)
    for i in range(max_sentence_length-current_sentence_length):
        tokenised_sentence.append('padding_term')
    
    # initialise sentece embedding matrix
    word_embedding_list = []
    unknown_word_count = 0
    for token in tokenised_sentence:
        try:
            word_vec = embedding_dict[token]
        
        except KeyError:
            word_vec = np.zeros((1,50))
            unknown_word_count = unknown_word_count + 1
            #print(token)
        reshaped_word_vec = word_vec.reshape((1,-1))
        word_embedding_list.append(reshaped_word_vec)
        
    # concat list to get matrix
    sentence_embedding = np.concatenate(word_embedding_list)
    #print(unknown_word_count)
    return sentence_embedding
    
#%%
def buildAllWordEmbeddings(data_frame,embedding_dict):
    """ Fucntion to generate the embeddings for each sentence and then stack
        them ready to be joined to the pos embed and fed to cnn"""
    list_of_sent_embeds = []
    for i,k in data_frame['TokenisedEmbed'].iteritems():
        sent_embed = buildWordEmbeddingSentence(k,embedding_dict)
        list_of_sent_embeds.append(sent_embed)
        

    word_embedding_tensor = np.stack(list_of_sent_embeds)
    return word_embedding_tensor



#%%
def buildDataforNetworks(train,test,word_embedding_dict):
    """train and test should be the dataframes produced by runParseAndPickle"""    
    # get arrays of distance vectos
    distance_train = np.concatenate(train.Distances.values,axis=0)
    distance_test= np.concatenate(test.Distances.values,axis=0)
    
    # get y vals
    train_y = train.IsInteraction.values.reshape((-1,1))
    test_y = test.IsInteraction.values.reshape((-1,1))
    train_y = to_categorical(train_y,2)
    test_y = to_categorical(test_y,2)
    
    # combine so that they can be passed to embedding as one array
    combined_distances = np.concatenate((distance_train,distance_test),axis=0)
    distance_embeddings = buildPositionEmbedding(combined_distances,5)
    
    # retrieve the train and test parts
    dist_embed_train = distance_embeddings[:len(distance_train),:,:]
    dist_embed_test = distance_embeddings[len(distance_train):,:,:]
    
    # make word embeddings
    word_embedding_train = buildAllWordEmbeddings(train,word_embedding_dict)
    word_embedding_test = buildAllWordEmbeddings(test, word_embedding_dict)
    
    # combine the word embeddings with distance embeddings
    word_plus_dist_train = np.concatenate((word_embedding_train,dist_embed_train),axis=-1)
    word_plus_dist_test = np.concatenate((word_embedding_test,dist_embed_test),axis=-1)

    train_x = np.expand_dims(word_plus_dist_train,-1)
    test_x = np.expand_dims(word_plus_dist_test,-1)

    with open('train_x.pickle', 'wb') as handle:
        pickle.dump(train_x, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('test_x.pickle', 'wb') as handle:
        pickle.dump(test_x, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('train_y.pickle', 'wb') as handle:
        pickle.dump(train_y, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('test_y.pickle', 'wb') as handle:
        pickle.dump(test_y, handle, protocol=pickle.HIGHEST_PROTOCOL)


#%%
def loadDataForNetworks():
    
    with open('train_x.pickle', 'rb') as handle:
        train_x = pickle.load(handle)
    
    with open('train_y.pickle', 'rb') as handle:
        train_y = pickle.load(handle)
    
    with open('test_x.pickle', 'rb') as handle:
        test_x = pickle.load(handle)
    
    with open('test_y.pickle', 'rb') as handle:
        test_y = pickle.load(handle)
    
    return train_x,train_y,test_x,test_y
    
#%%
def buildModel(filter_size):
    model = Sequential()
    model.add(Conv2D(10, (filter_size, filter_size), input_shape=(155, 60, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(2, activation='softmax'))    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model    

def runTestCNN(train_x,train_y,filter_size,num_eopchs):
    model = buildModel(filter_size)
    history = model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=num_eopchs, batch_size=32, verbose=2,shuffle=True)   
    pred = model.predict(test_x, batch_size=32, verbose=1)
    predicted = np.argmax(pred, axis=1)
    report = classification_report(np.argmax(test_y, axis=1), predicted)
    
    print(report)
    return predicted
# =============================================================================
#     with open("res_cnn_classifier.txt",'a') as handle:
#         handle.write("\n" + "filter size:"+str(filter_size)+"--num epochs:"+str(num_eopchs)+"\n"+report+"\n")
# =============================================================================

# =============================================================================
# embdic = buildWordEmbeddingsDict()
# train,test = loadTrainTest()
# buildDataforNetworks(train,test,embdic)
# =============================================================================

#train_x,train_y,test_x,test_y = loadDataForNetworks()


# =============================================================================
# TODO - Tidy up preprocessing of imbalanced classes
# TODO - make sure all target_substance replacements dont have trailing s etc
# TODO - make sure all tokens are lower case for retrieving emebeddings
# TODO - split al hypen-treated words up in preprocess
# =============================================================================

# =============================================================================
# runTestCNN(train_x,train_y,5,10)
# runTestCNN(train_x,train_y,6,10)
# runTestCNN(train_x,train_y,7,10)
# =============================================================================


#multi filter model
# =============================================================================
# branch1 = Sequential()
# branch1.add(Conv2D(10, (3,3), input_shape=(155, 60, 1), padding='same', activation='relu'))
# 
# branch2 = Sequential()
# branch2.add(Conv2D(10, (6,6), input_shape=(155, 60, 1), padding='same', activation='relu'))
# 
# branch3 = Sequential()
# branch3.add(Conv2D(10, (10,10), input_shape=(155, 60, 1), padding='same', activation='relu'))
# 
# branch4 = Sequential()
# branch4.add(Conv2D(10, (15,15), input_shape=(155, 60, 1), padding='same', activation='relu'))
# 
# branch5 = Sequential()
# branch5.add(Conv2D(10, (20,20), input_shape=(155, 60, 1), padding='same', activation='relu'))
# 
# branch6 = Sequential()
# branch6.add(Conv2D(10, (25,25), input_shape=(155, 60, 1), padding='same', activation='relu'))
# 
# model = Sequential()
# model.add(Merge([branch1, branch2, branch3,branch4,branch5,branch6], mode = 'concat'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.2))
# model.add(Flatten())
# model.add(Dense(2, activation='softmax'))
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# 
# history = model.fit([train_x,train_x,train_x,train_x,train_x,train_x], train_y, validation_data=([test_x,test_x,test_x,test_x,test_x,test_x], test_y), epochs=10, batch_size=32, verbose=2,shuffle=True)   
# pred = model.predict([test_x,test_x,test_x,test_x,test_x,test_x], batch_size=32, verbose=1)
# predicted = np.argmax(pred, axis=1)
# report = classification_report(np.argmax(test_y, axis=1), predicted)
# print(report)
# with open("res_cnn_multifilter_classifier.txt",'a') as handle:
#     handle.write("\nfilter size: 3,6,10,15,20,25   num epochs:10\n"+report+"\n")
# 
# =============================================================================




