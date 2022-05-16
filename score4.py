# Reference text recomedation(TFIDF and similarity) :https://towardsdatascience.com/build-a-text-recommendation-system-with-python-e8b95d9f251c
# version 3: add description code

import os
import torch
import torch.nn as nn
import json
import numpy as np
import os
import torch
from pathlib import Path
from transformers import BertTokenizerFast, BertForSequenceClassification
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from collections import Counter
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from nltk import word_tokenize, pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle
import nltk
from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk.stem.snowball import SnowballStemmer
import pandas as pd
from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
from azureml.core.model import Model
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')


def init():
    ##Global variable
    global classification_model,classification_tokenizer,classification_max_length ,target_names,STOPWORDS,df,summary_tokenizer,summary_model
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    # For multiple models, it points to the folder containing all deployed models (./azureml-models)

    #Text classification
    classification_main_path= Model.get_model_path('Class_fine_tune', version=6)
    classification_model_path =os.path.join(classification_main_path,'bbc-bert-base-uncased')
    target_names=[ 'business','entertainment','politics', 'sport', 'tech' ]   
    classification_model = BertForSequenceClassification.from_pretrained(classification_model_path, num_labels=len(target_names))
    classification_tokenizer = BertTokenizerFast.from_pretrained(classification_model_path)
    classification_max_length = 512

    #Text summarization
    summary_main_path= Model.get_model_path('summary_fine_tune', version=10)
    print(summary_main_path)
    summary_model_path =os.path.join(summary_main_path,'t5_small')
    print(summary_model_path)
    summary_tokenizer = AutoTokenizer.from_pretrained(summary_model_path)
    summary_model = AutoModelForSeq2SeqLM.from_pretrained(summary_model_path)

    # Recommenation text
    df=pd.read_pickle(os.path.join(classification_main_path,'df.pkl'))
    df.rename(columns={'article':'sentence'}, inplace=True)
    print(df.head(2))
    STOPWORDS = set(stopwords.words('english'))
   

def clean_text(text):
    ''' Cleaning text ,string to be lower case, remove punctuation and number.'''        
    text = text.lower()  # lowercase text
    text = re.sub('\n', ' ', text)
    text = re.sub('\t', ' ', text)
    text = re.sub("\'s",' ', text)
    # text = re.sub('[\d]', '', text)
    text = re.sub(r"[^\w\s]", ' ', text)
    return text

def pos_selected(text):
    '''Given a string of text, tokenize the text and select part of speech.'''
    # https://en.wikipedia.org/wiki/Brown_Corpus
    pos_selection=['CC','EX','FW','NN','NNS','NNP','NNPS','POS','PRP','PRP$','RP','WDT','WRB']
    tokenized = word_tokenize(text)
    selected_word = [word for (word, pos) in pos_tag(tokenized) if (pos in pos_selection) and (word not in STOPWORDS )]
    return selected_word 

def tokenizer(sentence, min_words=4, max_words=200, stemm:str=''):
    """ tokenize text , remove stop words and stemming words """
    sentence=sentence.lower()
    stemm=stemm.lower()
    tokenized= word_tokenize(sentence)
    #Use stemming word in tokenized sentence
    if stemm=='lemmatizer':
        Lemmatizer = WordNetLemmatizer()
        tokenized = [Lemmatizer.lemmatize(w) for w in tokenized]
    elif stemm=='snowball':
        Snowball = SnowballStemmer('english')
        tokenized = [Snowball.stem(w) for w in tokenized]
    elif stemm=='porter':
        Porter = PorterStemmer()
        tokenized = [Porter.stem(w) for w in tokenized]    
    else:
        tokenized =tokenized
    token = [w for w in tokenized if (len(w) > min_words and len(w) < max_words and (w not in STOPWORDS ))]
    return token    


def extract_best_indices(m, top_document, mask=None):
    #refer code 
    """
    Use sum of the cosine distance over all tokens.
    m (np.array): cos matrix of shape (nb_in_tokens, nb_dict_tokens)
    topk (int): number of indices to return (from high to lowest in order)
    """
    # return the sum on all tokens of cosinus for each sentence
    if len(m.shape) > 1:
        cos_sim = np.mean(m, axis=0) 
    else: 
        cos_sim = m
    # sorted index of cosine similarity from highest to smallest score 
    index = np.argsort(cos_sim)[::-1] 
    if mask is not None:
        assert mask.shape == m.shape
        mask = mask[index]
    else:
        mask = np.ones(len(cos_sim))
    #delete 0 cosine distance
    mask = np.logical_or(cos_sim[index] != 0, mask) 
    best_index = index[mask][:top_document]  
    return best_index



def get_recommendations(sentence, amount_document,columnname='heading' ):    
    """ get the document in order of highest cosine similarity relatively following the amont of document  """
    # Fit TFIDF    
    vectorizer = TfidfVectorizer( tokenizer=tokenizer) 
    tfidf_mat = vectorizer.fit_transform(df['sentence'].values)
    # tokenize input sentence
    tokens = [str(tok) for tok in tokenizer(sentence)]
    vec = vectorizer.transform(tokens)    
    # the matrix of cosine_similarity between sentence of document and database
    mat = cosine_similarity(vec, tfidf_mat)    
    # the index of document that have top cosine distance
    best_index = extract_best_indices(mat, top_document=amount_document)
    artical_list=df[columnname].tolist()
    type_artical_list=df['group_document'].tolist()
    all_list=[]
    if len(best_index)>0:
        for idx in best_index:            
            all_list.append({'Document heading':artical_list[idx].replace('\n',' ').strip(),'Document type':type_artical_list[idx]})
    else:
        all_list=[]
    # the detail of document that have top cosine distance
    return all_list
    
def BBC_classification_prediction(texts):
    texts=texts.split('\n')
    predict_list=[]
    for text in texts:
        # prepare our text into tokenized sequence
        inputs = classification_tokenizer(text, padding=True, truncation=True, max_length=classification_max_length, return_tensors="pt")
        # perform inference to our model
        outputs = classification_model(**inputs)
        # get output probabilities by doing softmax
        probs = outputs[0].softmax(1)
        # executing argmax function to get the candidate label
        predict_class=target_names[probs.argmax()]
        predict_list.append(predict_class)
    frequency_class=Counter(predict_list).most_common(1)
    return frequency_class[0][0]

def BBC_summarization(text,max_length_summarize):  
    preprocess_text = text.strip().replace("\n","")
    t5_prepared_Text = "summarize: "+preprocess_text
    # print ("Preprocessed and prepared text: \n", t5_prepared_Text)

    tokenized_text = summary_tokenizer.encode(t5_prepared_Text, return_tensors="pt").to('cpu')

    # summmarize 
    summary_ids = summary_model.generate(tokenized_text,
                                        num_beams=4,
                                        no_repeat_ngram_size=2,
                                        min_length=30,
                                        max_length=max_length_summarize,
                                        early_stopping=True)
                                        
    output = summary_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return output

def run(raw_data):
    try:        
        data = json.loads(raw_data)['data']
        sum_length = json.loads(raw_data)['length_summary']
        amount_recom = json.loads(raw_data)['amount_recom']
        print(data)
        # data='Spam e-mails tempt net shoppers'
        result = BBC_classification_prediction(data)
        # You can return any JSON-serializable object.                  
        result2=get_recommendations(data,amount_recom )
        result3=BBC_summarization(data,sum_length)
        return {'Document type':result ,'Document summary':result3,'Recommadation Document':result2}
    except Exception as e:
        error = str(e)
        return error
