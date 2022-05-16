# Reference code:https://github.com/huggingface/notebooks/blob/main/examples/summarization.ipynb

import torch
torch.cuda.empty_cache()
import argparse
from azureml.core import Run
import codecs
import os
import pandas as pd
import numpy as np
import torch
from transformers import Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from azureml.core.model import Model
from datasets import  load_metric
from transformers import AutoTokenizer
from datasets import Dataset
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import nltk
nltk.download('punkt')

def init():
    global device_ava,model_checkpoint,metric,tokenizer,model, max_input_length,max_target_length,batch_size
    device_ava = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Specify model name
    model_checkpoint ="t5-small"  
    # specify matrix for compute  
    metric = load_metric("rouge")    
    max_input_length = 1024
    max_target_length = 300
    batch_size = 8
    # Specify model
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    # Specify tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)    
    print('init')

def load_data(data_dir):   
    ''' load data from datastore into dataframe''' 
    li1=[]
    for fn in os.listdir(data_dir):
        path=os.path.join(data_dir,fn)
        for fn1 in os.listdir(path):
            path1=os.path.join(data_dir,fn,fn1)
            for fn2 in os.listdir(path1):
                path2=os.path.join(data_dir,fn,fn1,fn2)
                # open text
                f = codecs.open(path2, encoding='utf-8',errors='ignore')
                article1=list()
                for line in f:
                    if line !='\n':
                        article1.append(line)
                list_=[str(fn2),str(fn1),str(fn),str(article1[0]),str(' '.join(article1))]                
                df=pd.DataFrame([list_],columns=['filename','group_document','documents_type','heading','article'])
                li1.append(df)
    df_concat1=pd.concat(li1).reset_index().drop(columns=['index'])
    df_concat1['all']=df_concat1.apply(lambda x:x['group_document']+'_'+x['filename'],axis=1)
    print(df_concat1.shape)
    # dataframe of news article
    News=df_concat1[df_concat1['documents_type']=='News_Articles']
    # dataframe of summary
    Summaries=df_concat1[df_concat1['documents_type']=='Summaries']
    print('new shape:',News.shape)
    print(News.columns)
    print('Summaries:',Summaries.shape)
    print(Summaries.columns)
    Summaries1=Summaries[['all', 'article']]
    df=News.merge(Summaries1, on=['all'])
    df=df.rename(columns={'article_x':'document','article_y':'summary'})
    train,test1= train_test_split(df, test_size=0.3, random_state=39)
    validate,test= train_test_split(test1, test_size=0.5, random_state=39)
    print(df.shape)
    print(df.head())
    #train and validate dataset    
    sample_train_dataset = Dataset.from_dict({"document": train["document"].tolist(),"summary": train["summary"].tolist()})
    sample_valid_dataset = Dataset.from_dict({"document": validate["document"].tolist(),"summary": validate["summary"].tolist()})  
    print('load data')
    del df
    del train
    del test1
    del validate
    del News
    del Summaries
    del df_concat1
    return sample_train_dataset ,sample_valid_dataset,test

def compute_metrics(eval_pred):
    ''' compute the metrics'''
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)    
    # Rouge expects a newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]   
    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)    
    # Extract a few results
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    # Add mean generated length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)    
    return {k: round(v, 4) for k, v in result.items()}
    
def preprocess_function(examples):  
    if model_checkpoint in ["t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b"]:
        prefix = "summarize: "
    else:
        prefix = ""
    inputs = [prefix + doc for doc in examples["document"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["summary"], max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def pretrain_model(data_dir):
    print('pretrain_model')
    sample_train_dataset,sample_valid_dataset,test=load_data(data_dir)
    train_datasets = sample_train_dataset.map(preprocess_function, batched=True)
    valid_dataset = sample_valid_dataset.map(preprocess_function, batched=True)  
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    model_name = model_checkpoint.split("/")[-1]
    args = Seq2SeqTrainingArguments(
        f"{model_name}-finetuned-xsum",
        evaluation_strategy = "epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.05,
        save_total_limit=3,
        num_train_epochs=3,
        predict_with_generate=True,
        fp16=True,
        # push_to_hub=True,
    )
    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=train_datasets,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,    
    )
    print('trainer')
    trainer.train()
    model_path='./outputs/t5_small'
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)

if __name__ == "__main__":
    run = Run.get_context()
    parser = argparse.ArgumentParser()
    print('parser:',parser)
    parser.add_argument('--data_path',type=str, help='Path to the training data')
    # parser.add_argument('--output_dir', type=str, help='output directory')
    args = parser.parse_args()
    os.makedirs('outputs', exist_ok=True)
    print('args',args)
    print("===== DATA =====")
    print("DATA PATH: " + args.data_path)
    print(os.listdir(args.data_path))
    print("LIST FILES IN DATA PATH...")
    text_file = open("./outputs/Text_Class_name.txt", "w")
    text_file.write("Text Class name:"+ str(os.listdir(args.data_path))+'\n')
    text_file.write("News_Articles Class name:"+ str(os.listdir(os.path.join(args.data_path,'News_Articles')))+'\n')
    text_file.write("Summaries Class name:"+ str(os.listdir(os.path.join(args.data_path,'Summaries'))))
    text_file.close()
    init()
    pretrain_model(args.data_path)
    print("====1136======")
    print("================")