# Reference text classification code :https://github.com/plaban1981/Huggingface_transformers_course/blob/main/FineTuningBERT_Transformers.ipynb
# version 3: add description code
import argparse
import os
from azureml.core import Run
import codecs
import os
import codecs
import pandas as pd
import torch
from transformers import BertTokenizerFast, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



def load_data(data_dir):    
    li=[]
    data_dir=os.path.join(data_dir,'News_Articles')
    for group_document in os.listdir(data_dir):
            print(group_document)
            li1=[]
            for file_ in os.listdir(os.path.join(data_dir,group_document)):
                # print(file_)
                if file_[-3:]=='txt' :
                    #Open text use codecs library supporting text of utf-8
                    f = codecs.open(os.path.join(data_dir,group_document,file_), encoding='utf-8',errors='ignore')
                    article=[]
                    # read each line of text
                    for line in f:
                        if line !='\n':
                             article.append(line.replace('\n','').strip())
                list_=[file_,group_document,article[0],' '.join(article),article]             
                df=pd.DataFrame([list_],columns=['filename','group_document','heading','article','article_list'])
                li.append(df)
            df_concat1=pd.concat(li).reset_index().drop(columns=['index']) 
            li1.append(df_concat1)  
    df_concat=pd.concat(li1).reset_index().drop(columns=['index']) 
    df_concat['group_document_num']=df_concat['group_document'].astype("category").cat.codes
    file_path=os.path.join('./outputs/df.pkl')
    df_concat.to_pickle(file_path)
    list_all=[]
    # seperate article into sentence 
    for index, row in df_concat.iterrows():
        li=[]
        for line in row['article_list']:
            df1=pd.DataFrame([line],columns=['sentence'])
            li.append(df1)
        df_concat=pd.concat(li)
        df_concat['group_document']=row['group_document']
        list_all.append(df_concat)
    df_all=pd.concat(list_all)
    print(df_all.shape)
    print(df_all.head())
    print(df_all.tail())
    # convert name of group document into float category
    df_all['group_document_num']=df_all['group_document'].astype("category").cat.codes
    #seperate train test dataset
    train,test1= train_test_split(df_all, test_size=0.3, random_state=39)
    validate,test= train_test_split(test1, test_size=0.5, random_state=39)
    target_names=list(set(df_all['group_document'].tolist()))
    return train,validate,test,target_names

def dataframe_to_list(dataframe):
    '''convert datarame to list'''
    text=dataframe['sentence'].tolist()
    labels=dataframe['group_document_num'].tolist()
    return text,labels

class NewsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor([self.labels[idx]])
        return item
    def __len__(self):
        return len(self.labels)

def preparing_data(data_dir,tokenizer,max_length):
    # load dataset
    train,validate,test,target_names=load_data(data_dir)
    train_texts, train_labels=dataframe_to_list(train)
    valid_texts,  valid_labels=dataframe_to_list(validate)
    # tokenize trian dataset and validation dataset 
    # truncate the sentence  while the words of sentence is more than max_length
    # pad with 0 the sentence  while the words of sentence is less than max_length
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=max_length)
    valid_encodings = tokenizer(valid_texts, truncation=True, padding=True, max_length=max_length)
    # convert our tokenized data into a torch Dataset
    train_dataset = NewsDataset(train_encodings, train_labels)
    valid_dataset = NewsDataset(valid_encodings, valid_labels)
    return train_dataset,valid_dataset,test,target_names

def compute_metrics(pred):
  # the actual label
  labels = pred.label_ids
  # prediction label
  preds = pred.predictions.argmax(-1)
  # calculate accuracy 
  acc = accuracy_score(labels, preds)
  return {
      'accuracy': acc,
  }
def pretrain_model(data_dir):
    device_ava = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
    # the model use base uncased BERT
    model_name = "bert-base-uncased"
    # max sequence length 
    max_length = 512
    #text tokenizing use BertTokenizerFast
    tokenizer = BertTokenizerFast.from_pretrained(model_name, do_lower_case=True)
    train_dataset,valid_dataset,test,target_names=preparing_data(data_dir,tokenizer,max_length)
    # load the model using GPU
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=len(target_names)).to(device_ava)
    training_args = TrainingArguments(
        output_dir='./results',          # output directory
        num_train_epochs=3,              # total number of training epochs
        per_device_train_batch_size=8,  # batch size per device during training
        per_device_eval_batch_size=20,   # batch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        load_best_model_at_end=True,     # load the best model when finished training (default metric is loss)
        logging_steps=400,               # log & save weights each logging_steps
        save_steps=400,
        evaluation_strategy="steps",     # evaluate each logging_steps
    )
    trainer = Trainer(
        model=model,                         # the Transformers model to be trained
        args=training_args,                  # training arguments
        train_dataset=train_dataset,         # training dataset
        eval_dataset=valid_dataset,          # evaluation dataset
        compute_metrics=compute_metrics,     # the callback that computes metrics 
    )
    #train the model
    trainer.train()
    # evaluate the model 
    trainer.evaluate()
    # saving the fine tuned model & tokenizer
    model_path1='./outputs/bbc-bert-base-uncased'
    model.save_pretrained(model_path1)
    tokenizer.save_pretrained(model_path1)  
                            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    print('parser:',parser)
    parser.add_argument('--data_path',type=str, help='Path to the training data')
    args = parser.parse_args()
    os.makedirs('outputs', exist_ok=True)
    print('args',args)
    print("===== DATA =====")
    print("DATA PATH: " + args.data_path)
    print(os.listdir(args.data_path))
    print("LIST FILES IN DATA PATH...")
    #Add text file of class name
    text_file = open("./outputs/Text_Class_name.txt", "w")
    text_file.write("Text Class name:"+ str(os.listdir(args.data_path))+'\n')
    text_file.write("News_Articles Class name:"+ str(os.listdir(os.path.join(args.data_path,'News_Articles')))+'\n')
    text_file.write("Summaries Class name:"+ str(os.listdir(os.path.join(args.data_path,'Summaries'))))
    pretrain_model(args.data_path)
    print("================")