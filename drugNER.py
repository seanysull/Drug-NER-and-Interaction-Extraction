# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 11:25:15 2018

@author: sean.o.sullivan
"""
import nltk
import xml.etree.ElementTree as ET
import pandas as pd
import os
from nltk.tokenize import TreebankWordTokenizer
import numpy as np
import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
import sklearn
import scipy.stats
from sklearn.metrics import make_scorer
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import RandomizedSearchCV
from datetime import datetime
import string
import pickle
import eli5
#%%
# =============================================================================
# tree2 = ET.parse('Data/Train/MedLine/11160777.xml')
# root2 = tree2.getroot()
# sents2 = list(root2)
# s1=sents2[7]
# ents = s1.findall("entity")
# text=s1.attrib['text']
# =============================================================================


#%%
def getTokensAndOffsets(sentence):
    """
    Given a sentence, as a string tokenise it and return tokens and token spans as a list of tuples
    """
    tokens = TreebankWordTokenizer().tokenize(sentence)
    offsets = TreebankWordTokenizer().span_tokenize(sentence)
    tokens_and_offsets = list(zip(tokens,offsets))
    return tokens_and_offsets
    
#%%
def getFollowLabel(label):
    """
    function to return the I version of a B label - thats is given the label for a
    beginning label return the In label that follows
    """
    In_labels = {'B-brand': 'I-brand',
                    'B-drug': 'I-drug' ,
                     'B-group': 'I-group',
                     'B-drug_n': 'I-drug_n',
                 }
    
    return In_labels[label]
#%%
def assignLabel(index_token_offset, offset_label_dict):
    """
    function to assign labels to sequences of contiguous drug names, index_token_offset
    always has the final offset wrong by +1 hence the offset[1]-1
    """
    #creat empty list ready to append word plus label
    token_plus_biolabel = []
    # get list of keys(offsets) from offset-drug_type dict these are the loactions of the entities
    keys = list(offset_label_dict)
    
    # flag to say we are in a sequence ie a multi word drug name
    in_sequence_flag = 0    
    
    for (index,token,offset) in index_token_offset:
        # we delete the entity indexes(keys) as we go so we check if we have deleted all keys
        # if so we need to label all remaining terms as 'O' -  other
        if len(keys) != 0:
            k1=keys[0][0]
            k2=keys[0][1]
        
        elif len(keys) == 0:
            token_plus_biolabel.append((token,'O'))
            continue
            
        # check if we are in sequence, if not proceed as normal if so check if it is end or middle
        if in_sequence_flag == 0:
        # if current start of token offset is less than that of current key(entity)
        # then assign other label and continue
            if k1>offset[0]:
                token_plus_biolabel.append((token,'O'))
            
            elif k1 == offset[0] and k2 == offset[1]-1:    
                # get label then append token and label to list
                label = offset_label_dict[(k1,k2)]                
                token_plus_biolabel.append((token,label))
                # delete the matching key as it is no longer needed
                del keys[0]
                
            
            elif k1 == offset[0] and k2 > offset[1]-1:
                 # get label for first token in multi worder
                 label1 = offset_label_dict[(k1,k2)]
                 token_plus_biolabel.append((token,label1))
                 in_sequence_flag = 1
                 
        else:
             # check if word is the terminal of a sequence:
             # if so delete key and change flag  otherwise just label and continue
            if k2 == offset[1]-1:
                label_init = offset_label_dict[(k1,k2)]
                label_follow = getFollowLabel(label_init)
                token_plus_biolabel.append((token,label_follow))
                in_sequence_flag = 0
                del keys[0]
             # token is in middle of sequence so just add follow label and proceed to next word    
            else:
                label_init = offset_label_dict[(k1,k2)]
                label_follow = getFollowLabel(label_init)
                token_plus_biolabel.append((token,label_follow))
                 
        
    return token_plus_biolabel    

def buildDrugList():
    """build the list of drugnames from db data"""
    tree = ET.parse('C:/Users/sean.o.sullivan/Documents/uni/ANLP/drugbank_all_full_database.xml/fulldatabase.xml')
    drugnames = set([name.text for name in tree.iter("{http://www.drugbank.ca}name")])
    with open('drugbank_names.pickle','wb') as handle:
        pickle.dump(drugnames,handle,protocol=pickle.HIGHEST_PROTOCOL)
		
def loadDrugList():
    with open('drugbank_names.pickle', 'rb') as handle:
        drugnames = pickle.load(handle)		
    return drugnames
	
	
def buildWordFeatures(token_pos_bio, word_index, druglist):
    """Function to build the dictionary of features for a given word in a given sentence, expects a list of (word, pos) tuples
       or (word, pos, bio label) tuples, the annotated sentence, and the index of the desired word(tuple) """
    def hasNumbers(inputString):
        return any(char.isdigit() for char in inputString)
    
    word = token_pos_bio[word_index][0]    
    pos_tag = token_pos_bio[word_index][1] 
    drugnames = druglist
    
    word_features = {
    # features for all words in the sentence
        "bias": 1.0,
        "word": word.lower(),
        "pos_tag": pos_tag,
        "pos_tag_first2": pos_tag[:2],
        "suffix_2": word[-2:],
        "suffix_3": word[-3:],
        "suffix_4": word[-4:],
        "suffix_5": word[-5:],
        "prefix_2": word[:2],
        "prefix_3": word[:3],
        "prefix_4": word[:4],
        "prefix_5": word[:5],
        "IsBegin": word_index == 0,
        "IsEnd": word_index == len(token_pos_bio)-1,
        "All_Upper": word.isupper(),
        "ContainsNumbers": hasNumbers(word),
	   	 "AppersInDrugBank": word in drugnames,
		 "ContainsHyphen": "-" in word,
        "WordLen": len(word),
        "Capitalised": word[0].isupper()
    }
    
    if word_index > 0:
 
        word_before = token_pos_bio[word_index - 1][0]
        pos_tag_before = token_pos_bio[word_index - 1][1]
        word_features.update({
                "pos_tag_before": pos_tag_before,
                "pos_tag_before_f2": pos_tag_before[:2],
                "word_before": word_before.lower(),
                "All_Upper_before": word_before.isupper(),
                "suffix_2_before": word_before[-2:],
                "suffix_3_before": word_before[-3:],
                "suffix_4_before": word_before[-4:],
                "suffix_5_before": word_before[-5:],
                "prefix_2_before": word_before[:2],
                "prefix_3_before": word_before[:3],
                "prefix_4_before": word_before[:4],
                "prefix_5_before": word_before[:5],
                "AppersInDrugBank": word_before in drugnames
        })

    if word_index < len(token_pos_bio)-1:          

        word_after = token_pos_bio[word_index + 1][0]
        pos_tag_after = token_pos_bio[word_index + 1][1]
        word_features.update({
                "pos_tag_after": pos_tag_after,
                "pos_tag_after_f2": pos_tag_after[:2],
                "word_after": word_after.lower(),
                "All_Upper_after": word_after.isupper(),
                "suffix_2_after": word_after[-2:],
                "suffix_3_after": word_after[-3:],
                "suffix_4_after": word_after[-4:],
                "suffix_5_after": word_after[-5:],
                "prefix_2_after": word_after[:2],
                "prefix_3_after": word_after[:3],
                "prefix_4_after": word_after[:4],
                "prefix_5_after": word_after[:5],
                "AppersInDrugBank": word_after in drugnames
        })
    
    return word_features

def buildSentenceFeatures(token_pos_bio, druglist):
    return [buildWordFeatures(token_pos_bio, index, druglist) for index in range(len(token_pos_bio))]
    

#%% 
def parseNER(pathtoxml):
    """Fucntion to parse the entities expects to be passed a path to the xml file to parse"""
    #parse tree using Elementtree
    tree = ET.parse(pathtoxml)
    #get the root of the tree
    root = tree.getroot()

    dataEnts = []
    
    sents = [elem for elem in tree.iter("sentence")]
    
    DocumentId = root.attrib.get("id")
    druglist = loadDrugList()
    # TODO modify to include tags for compound drugs (only 33 entities in 22 sentences )
    #labels for assigning correct BO tag to words in sentences
    dtype_labels = {'brand':'B-brand',
                    'drug':'B-drug',
                     'group' :'B-group',
                     'drug_n' :'B-drug_n',
                     'other':'O'}
    #import ipdb; ipdb.set_trace()
    for sentence in sents:
        # get sentence id and text
        sentId = sentence.attrib.get("id")
        senText = sentence.attrib.get("text")
        #check that sentence has children(entities) if not just add sentence and id
        entities = sentence.findall("entity")
        #get the tokens, offsets, pos tags in various combinations
        tokens_and_offsets  = getTokensAndOffsets(senText)
        just_tokens = [token for (token, offset) in tokens_and_offsets]
        just_offsets = [offset for (token, offset) in tokens_and_offsets]
        indexes = list(range(len(just_tokens)))
        index_token_offset = list(zip(indexes,just_tokens,just_offsets))
        tokens_plus_POS = nltk.pos_tag(just_tokens)
        entity_dictionaries = [entity.attrib for entity in entities]
        # if entities is empty then sentence contains no drugs and we pass empty dict to get all'O' tags
        
        if entities == []:
            offset_entity_dict = {}
        
        # if ntities is not empty proceed to add tags for words as appropriate
        elif entities != []:
                        
            offset_entity_dict = {}
            for entity in entities:
                # build dictionary of offset:entity type for all entities in sentence
                offsetstring = entity.attrib.get('charOffset')
                # split offset string on the hyphen, if length is > 3 then ignore as it is difficult
                # and rare edge case
                offset = offsetstring.split("-")
                if len(offset)>2: 
                    continue
                # convert to tuple of integers
                offset_start = int(offset[0])
                offset_end = int(offset[1])
                offset_full = (offset_start,offset_end)
                # get drug type and correct bio label
                drugtype = entity.attrib.get('type')
                biolabel = dtype_labels[drugtype]
                
                offset_entity_dict[offset_full] = biolabel
                
                
            # using offset:label dict go through sentence and assign label if offset matches dict or assign 'O' otherwise
        token_plus_biolabel = assignLabel(index_token_offset,offset_entity_dict)
        
        # get just labels for y vals in training
        just_bio_label = [label for (token,label) in token_plus_biolabel]
                
            # collect the words, pos tags and bio labels       
        token_pos_bio = [(token,pos,bio) for (token,bio),(_,pos) in zip(token_plus_biolabel,tokens_plus_POS)]
        
        features = buildSentenceFeatures(token_pos_bio,druglist)
        
        datarow = {"DocumentId":DocumentId,"SentenceId":sentId,
                   "Sentence":senText,"LabeledToken":token_plus_biolabel,
                   "TokenPosBio":token_pos_bio,"Features": features, 
                   "TargetLabels": just_bio_label, "Entities": entity_dictionaries}
            
        dataEnts.append(datarow)
                    
    dfEnts = pd.DataFrame(dataEnts)
    
    return dfEnts
#%%

def buildTrainTestNER():
    """collect the full filenames of the drugbank/medline train/test where test is split into NER and DDI subtasks
    and build the dataframes, drugbank and medline are separate in case we wish to compare results against eachother"""
    
    # DRUGBANK TRAINING SET -----------------------------------------------------------------------
    pathtrainDB = 'C:\\Users\\sean.o.sullivan\\Documents\\uni\\ANLP\\Data\\Train\\DrugBank'
    filestrainDB = os.listdir(pathtrainDB)
    filepathstrainDB = [os.path.join(pathtrainDB,f) for f in filestrainDB]
    dflistTrainDBent = []
    for file in filepathstrainDB:
        dfents = parseNER(file)
        print("parsed: "+file)
        #append entity df to entities and pairs to pairs
        dflistTrainDBent.append(dfents)
    # concat dfs together and pickle file for drugbank entities and pairs   
    dfTrainDBents = pd.concat(dflistTrainDBent)
    dfTrainDBents.to_pickle('C:\\Users\\sean.o.sullivan\\Documents\\uni\\ANLP\\Data\\Train\\dfDBentsTrain.pkl')
       
    # MEDLINE TRAINING SET -----------------------------------------------------------------------
    pathtrainML = 'C:\\Users\\sean.o.sullivan\\Documents\\uni\\ANLP\\Data\\Train\\MedLine'
    filestrainML = os.listdir(pathtrainML)
    filepathstrainML = [os.path.join(pathtrainML,f) for f in filestrainML]
    dflistTrainMLent = []
    for file in filepathstrainML:
        dfents = parseNER(file)
        #append entity df to entities and pairs to pairs
        dflistTrainMLent.append(dfents)
     #concat dfs together and pickle file for drugbank entities and pairs   
    dfTrainMLents = pd.concat(dflistTrainMLent)
    dfTrainMLents.to_pickle('C:\\Users\\sean.o.sullivan\\Documents\\uni\\ANLP\\Data\\Train\\dfMLentsTrain.pkl')
        
    # DRUGBANK NER TEST SET -----------------------------------------------------------------------
    pathtestDBNER = 'C:\\Users\\sean.o.sullivan\\Documents\\uni\\ANLP\\Data\\Test\\TestNER\\DrugBank'
    filestestDBNER = os.listdir(pathtestDBNER)
    filepathstestDBNER = [os.path.join(pathtestDBNER,f) for f in filestestDBNER]
    dflisttestDBNER = []
    
    for file in filepathstestDBNER:
        dfs = parseNER(file)
        #append entity df to entities and pairs to pairs
        dflisttestDBNER.append(dfs)
     #concat dfs together and pickle file for drugbank entities and pairs   
    dftestDBNER = pd.concat(dflisttestDBNER)
    dftestDBNER.to_pickle('C:\\Users\\sean.o.sullivan\\Documents\\uni\\ANLP\\Data\\Test\\dfDBNERTest.pkl')
    
    # MEDLINE NER TEST SET -----------------------------------------------------------------------
    pathtestMLNER= 'C:\\Users\\sean.o.sullivan\\Documents\\uni\\ANLP\\Data\\Test\\TestNER\\MedLine'
    filestestMLNER = os.listdir(pathtestMLNER)
    filepathstestMLNER = [os.path.join(pathtestMLNER,f) for f in filestestMLNER]
    dflisttestMLNER = []
    
    for file in filepathstestMLNER:
        dfs = parseNER(file)
        #append entity df to entities and pairs to pairs
        dflisttestMLNER.append(dfs)
     #concat dfs together and pickle file for drugbank entities and pairs   
    dftestMLNER = pd.concat(dflisttestMLNER)
    dftestMLNER.to_pickle('C:\\Users\\sean.o.sullivan\\Documents\\uni\\ANLP\\Data\\Test\\dfMLNERTest.pkl')
   
#%%

def prepareTrainTestforTraining():
    """
    small function to load the pickled datasets concatenate then extract the appropriate columns
    """
    drugbank_train = pd.read_pickle('C:\\Users\\sean.o.sullivan\\Documents\\uni\\ANLP\\Data\\Train\\dfDBentsTrain.pkl')    
    medline_test = pd.read_pickle('C:\\Users\\sean.o.sullivan\\Documents\\uni\\ANLP\\Data\\Train\\dfMLentsTrain.pkl')
    train = pd.concat([drugbank_train,medline_test])
    train.reset_index(drop=True,inplace=True)
    train_x = train['Features']
    train_y = train['TargetLabels']
    
    drugbank_test = pd.read_pickle('C:\\Users\\sean.o.sullivan\\Documents\\uni\\ANLP\\Data\\Test\\dfDBNERTest.pkl')
    medline_test = pd.read_pickle('C:\\Users\\sean.o.sullivan\\Documents\\uni\\ANLP\\Data\\Test\\dfMLNERTest.pkl')
    test = pd.concat([drugbank_test,medline_test])
    test.reset_index(drop=True,inplace=True)
    test_x = test['Features']
    test_y = test['TargetLabels']
    
    return train_x, train_y, test_x, test_y, test
#%%
#parseNER('Data/Train/DrugBank/Aciclovir_ddi.xml')  
#buildTrainTestNER()
             
if __name__ == "__main__":
     
    train_x, train_y, test_x, test_y, testfull = prepareTrainTestforTraining()
    
    crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs', 
    c1=0.1, 
    c2=0.1, 
    max_iterations=10, 
    all_possible_transitions=True
    )
    
    crf.fit(train_x, train_y)    
    weight_explined = eli5.format_as_text(eli5.explain_weights(crf, top=30))
    
    
    labels = list(crf.classes_)
    labels.remove('O')
    labels
    y_pred = crf.predict(test_x)
    
    sorted_labels = sorted(labels, key=lambda name: (name[1:], name[0]))
    

    print(metrics.flat_classification_report(test_y, y_pred, labels=sorted_labels, digits=3))
    classification_report = metrics.flat_classification_report(test_y, y_pred, labels=sorted_labels, digits=3)
    print(classification_report)
    
# =============================================================================
#     with open('NERresults.txt','a') as res:
#         res.write('\n'+str(datetime.now())+'\n'+classification_report)#'\n\n'+ weight_explined)
#         
# =============================================================================
# =============================================================================
#     compare = pd.DataFrame(test_y)
#     compare['pred'] = y_pred
#     compare['text'] = testfull['Sentence']
#     compare['entities'] = testfull['Entities']
#     compare_wrong = compare[compare['pred'] != compare['TargetLabels']]
# =============================================================================

