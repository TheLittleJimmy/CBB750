# -*- coding: utf-8 -*-

epochs = 20
max_len = 300

# Experiment_name
experiment_name = 'basline_entire_dataset'

# Naming: model
model_path = '/content/drive/My Drive/CBB-750/Final Project/model/' + experiment_name + '.pt'
print(model_path)

# Naming: training history
training_df_path = '/content/drive/My Drive/CBB-750/Final Project/history/' + experiment_name + '.csv'
print(training_df_path)

"""### GPU"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor

from typing import Tuple, List

import random
import math
import os
import time
import json
import numpy as np
from collections import Counter



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Pytorch version is: ", torch.__version__)
print("You are using: ", DEVICE)

# mount Google Drive 
from google.colab import drive
drive.mount('/content/drive')

"""### Read Dataset"""

# load the file into a dataframe
import pandas as pd

IG_P = pd.read_excel('/content/drive/My Drive/CBB-750/Final Project/Dataset.xlsx', sheet_name='Info Giving - Patient')
IG_C = pd.read_excel('/content/drive/My Drive/CBB-750/Final Project/Dataset.xlsx', sheet_name='Info Giving - Clinician')
IG_A = pd.read_excel('/content/drive/My Drive/CBB-750/Final Project/Dataset.xlsx', sheet_name='Info Giving - Auto')

IS_C = pd.read_excel('/content/drive/My Drive/CBB-750/Final Project/Dataset.xlsx', sheet_name='Info Seek - Clinician')
IS_P = pd.read_excel('/content/drive/My Drive/CBB-750/Final Project/Dataset.xlsx', sheet_name='Info Seek - Patient')

Emo_P = pd.read_excel('/content/drive/My Drive/CBB-750/Final Project/Dataset.xlsx', sheet_name='Emotion - Patient')
Emo_C = pd.read_excel('/content/drive/My Drive/CBB-750/Final Project/Dataset.xlsx', sheet_name='Emotion - Clinician')

P_C = pd.read_excel('/content/drive/My Drive/CBB-750/Final Project/Dataset.xlsx', sheet_name='Partnership - Clinician')
P_P = pd.read_excel('/content/drive/My Drive/CBB-750/Final Project/Dataset.xlsx', sheet_name='Partnership - Patient')

# Combine IG_P, IG_C, IG_A into one dataframe
IG = pd.concat([IG_P, IG_C, IG_A], axis=0, ignore_index=True)
IG
# Combine IS_C, IS_P into one dataframe
IS = pd.concat([IS_C, IS_P], axis=0, ignore_index=True)

# Combine Emo_P, Emo_C into one dataframe
Emo = pd.concat([Emo_P, Emo_C], axis=0, ignore_index=True)

# Combine P_C, P_P into one dataframe
P = pd.concat([P_C, P_P], axis=0, ignore_index=True)

# Add labels to each dataframe
IG['Label'] = 'Info Giving'
IS['Label'] = 'Info Seeking'
Emo['Label'] = 'Emotion'
P['Label'] = 'Partnership'

# Combine all dataframes into one dataframe
df = pd.concat([IG, IS, Emo, P], axis=0, ignore_index=True)

# import package and torch
!pip install --upgrade packaging --quiet
!pip install torch --quiet
import torch

assert torch.cuda.is_available()

# Tell torch to use GPU
device = torch.device("cuda")
print('Running GPU: {}'.format(torch.cuda.get_device_name()))

possible_labels = df.Label.unique()

label_dict = {}
for index, possible_label in enumerate(possible_labels):
    label_dict[possible_label] = index
label_dict

df['label'] = df.Label.replace(label_dict)
df['text'] = df['Quotation Content']

import seaborn as sns
sns.histplot(df['Label'])

"""### Data Split"""

from sklearn.model_selection import train_test_split

df_train, df_rem = train_test_split(df,
                                    test_size=0.3,
                                    random_state=1,
                                    stratify=df['label'])

df_val, df_test = train_test_split(df_rem, test_size=0.5, random_state=1,
                                    stratify=df_rem['label'])

for item in df_train, df_val, df_test:
    print('Shape: {}'.format(item.shape))

# Show df_train class distribution 
sns.histplot(df_train['Label'])   
sns.histplot(df_val['Label'])
sns.histplot(df_test['Label'])

"""### Tokenization"""

!pip install transformers --quiet

import transformers
from transformers import BertTokenizer

# Load BERT tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name, 
                                          do_lower_case=True)

def get_encoded_dict(df):
    input_ids = []
    attention_mask = []

    for text in df['Quotation Content']:
        encoded = tokenizer.encode_plus(text,
                                        add_special_tokens=True,
                                        padding='max_length',
                                        return_attention_mask=True,
                                        max_length=max_len,
                                        return_tensors='pt',
                                        truncation=True)

        input_ids.append(encoded['input_ids'])
        attention_mask.append(encoded['attention_mask'])
        
    return input_ids, attention_mask

# Cat lists to tensors for TensorDataset
def get_tensors(input_ids, attention_mask):
    
    input_ids = torch.cat(input_ids, dim=0)
    attention_mask = torch.cat(attention_mask, dim=0)
    
    return input_ids, attention_mask

# Import tools for Dataloader
from torch.utils.data import TensorDataset,DataLoader,RandomSampler,SequentialSampler

# Convert df to DataLoader
def get_dataloader(df, batch_size=32):
    temp_ids, temp_masks = get_encoded_dict(df)
    
    # Convert to tensors
    temp_ids, temp_masks = get_tensors(temp_ids, temp_masks)
    temp_labels = torch.tensor(df['label'].values)
    
    # Generate dataset
    temp_dataset = TensorDataset(temp_ids,
                                 temp_masks,
                                 temp_labels)
    
    # Generate dataloader
    temp_dataloader = DataLoader(temp_dataset,
                                 batch_size=batch_size,
                                 sampler=RandomSampler(temp_dataset))
    
    return temp_dataloader

# Get dataloader for all dataframes
train_dataloader = get_dataloader(df_train)
val_dataloader = get_dataloader(df_val)
test_dataloader = get_dataloader(df_test)

"""### Import BERT"""

from transformers import BertForSequenceClassification, AdamW, BertForPreTraining

model = BertForSequenceClassification.from_pretrained(model_name,
                                                      num_labels=len(label_dict),
                                                      output_attentions=False,
                                                      output_hidden_states=False)

# Tell mode to use CUDA
model.cuda()

# Configuring optimizer
optimizer = AdamW(model.parameters(),
                  lr = 3e-5)

# Configuring scheduler
from transformers import get_linear_schedule_with_warmup

# Total steps: number of batchers * epochs
total_steps = len(train_dataloader) * epochs

# Set up the scheduler
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=10,
                                            num_training_steps=total_steps)

"""### Training"""

# Import materics
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import time

import numpy as np
# Reset history lists
training_stats = []

for epoch_i in range(epochs):
    
    # timer_start
    epoch_t0 = time.time()
    
    print('【EPOCH: {}/ {}】'.format(epoch_i+1, epochs))
    print('Trainig Phase')
    
    # Set training mode
    model.train()
    
    # Reset training loss
    total_training_loss = 0
    
    # Batch and forward
    for batch in train_dataloader:
        b_input_ids = batch[0].to(device)
        b_masks = batch[1].to(device)
        b_labels = batch[2].to(device)
    
        # Reset gradients before 
        model.zero_grad()
        
        # Forward pass
        res = model(b_input_ids,
                    token_type_ids=None,
                    attention_mask=b_masks,
                    return_dict=True,
                    labels=b_labels)
        
        loss = res.loss
        logits = res.logits
        
        # sumup training loss
        total_training_loss += loss.item()
        
        # backpropagation
        loss.backward()
        
        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # update optimizer and scheduler
        optimizer.step()
        scheduler.step()
        
    # averrage loss
    avg_train_loss = total_training_loss/len(train_dataloader)
    print("  Average training loss: {0:.4f}".format(avg_train_loss))
    
    
    # validation
    print('Validation Phase')
    
    # Reset validation loss
    total_val_loss = 0
    
    # Set up lists
    ls_val_logits = []
    ls_val_labels = []

    # Get batchs from val_dataloader
    for batch in val_dataloader:
        b_input_ids = batch[0].to(device)
        b_masks = batch[1].to(device)
        b_labels = batch[2].to(device)

        # No need to calculate gradients
        with torch.no_grad():

            res = model(b_input_ids,
                        token_type_ids=None,
                        attention_mask=b_masks,
                        labels=b_labels,
                        return_dict=True)

        val_loss = res.loss
        val_logits = res.logits
        total_val_loss += val_loss.item()

        # Convert logitis to numpy format
        val_logits = np.argmax(val_logits.cpu().detach().numpy(), axis=1)
        val_labels = b_labels.cpu().detach().numpy()

        # Append data to the lists
        for logit in val_logits:
            ls_val_logits.append(logit)

        for label in val_labels:
            ls_val_labels.append(label)
    
    # Get accuracy score and val_loss
    acc = accuracy_score(ls_val_logits, ls_val_labels)
    avg_val_loss = total_val_loss/len(val_dataloader)
    
    # Print out validation performance
    print('  Average validation loss: {:.4f}'.format(avg_val_loss))
    print('  Validation accruacy: {:.4f}'.format(acc))
    
    
    # timer_end
    epoch_time_spent = time.time() - epoch_t0
    print('  Time spent on the epoch: {:.2f}'.format(epoch_time_spent))
    print('\n')
    
    # Recording training stats
    training_stats.append(
        {
            'Epoch': epoch_i+1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'val_acc': acc,
            'time_spent': epoch_time_spent
        })

"""### Training performance"""

train_loss = []
val_loss = []
val_acc = []
time_spent = []

for i in range(len(training_stats)):
    train_loss.append(training_stats[i]['train_loss'])
    val_loss.append(training_stats[i]['val_loss'])
    val_acc.append(training_stats[i]['val_acc'])
    time_spent.append(training_stats[i]['time_spent'])

# Print time spent
print('Time spent on training {} epochs: {:.0f}'.format(epochs, np.sum(time_spent)/60) + ' minutes')

# Plotting
import matplotlib.pyplot as plt

plt.figure(figsize=(10,3))

plt.subplot(1,2,1)
plt.plot(train_loss)
plt.plot(val_loss)
plt.title('loss')

plt.subplot(1,2,2)
plt.plot(val_acc, color='red', linestyle='--')
plt.ylim(0.2, 0.7)
plt.title('accuracy')

torch.save(model.state_dict(), model_path)

"""### Perform on the test data"""

# Prepare df_test for prediction
t_input_ids, t_attention_mask = get_encoded_dict(df_test)
t_input_ids, t_attention_mask = get_tensors(t_input_ids, t_attention_mask)

# Prepare dataset and dataloader
test_dataset = TensorDataset(t_input_ids, t_attention_mask)
test_dataloader = DataLoader(test_dataset,
                             batch_size=32,
                             sampler=SequentialSampler(test_dataset))

# Show dataloader length
print('Number of batches in the dataloader: {}'.format(len(test_dataloader)))

# Setup lists for predictions and labels
ls_test_pred = []

# Get batchs from test_dataloader
for batch in test_dataloader:
    b_input_ids = batch[0].to(device)
    b_masks = batch[1].to(device)
    
    with torch.no_grad():

        res = model(b_input_ids,
                    attention_mask=b_masks,
                    return_dict=True)

        test_logits = res.logits
        test_logits = np.argmax(test_logits.cpu().detach().numpy(), axis=1)
        
        for pred in test_logits:
            ls_test_pred.append(pred)

# Set up list of test labels
ls_test_labels = df_test['Label'].values
rev_subs = { v:k for k,v in label_dict.items()}
ls_test_pred = [rev_subs.get(item,item)  for item in ls_test_pred]
# Get accuracy score and val_loss
acc = accuracy_score(ls_test_pred, ls_test_labels)
print('Prediction accuracy: {:.4f}'.format(acc))

label_dict.keys()

# Confusion matrix
cm = confusion_matrix(ls_test_labels, ls_test_pred)
labels = ["Emotion","Info Giving","Info Seeking","Partnership"]

fig = plt.figure(figsize=(8,8))
ax= fig.add_subplot(1,1,1)
sns.heatmap(cm, annot=True, cmap="Greens",ax = ax,fmt='g'); #annot=True to annotate cells

# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(labels); ax.yaxis.set_ticklabels(labels);
plt.setp(ax.get_yticklabels(), rotation=30, horizontalalignment='right')
plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')     
plt.show()
print(classification_report(list(ls_test_labels),list(ls_test_pred),labels=list(labels)))

"""### ClinicalBERT"""

!pip install pytorch-pretrained-bert

from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
model = BertForSequenceClassification.from_pretrained(model_name,
                                                      num_labels=len(label_dict),
                                                      output_attentions=False,
                                                      output_hidden_states=False)

# Tell mode to use CUDA
model.cuda()

"""### Tokenization"""

!pip install transformers --quiet

import transformers
from transformers import BertTokenizer

# Load BERT tokenizer
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

def get_encoded_dict(df):
    input_ids = []
    attention_mask = []

    for text in df['Quotation Content']:
        encoded = tokenizer.encode_plus(text,
                                        add_special_tokens=True,
                                        padding='max_length',
                                        return_attention_mask=True,
                                        max_length=max_len,
                                        return_tensors='pt',
                                        truncation=True)

        input_ids.append(encoded['input_ids'])
        attention_mask.append(encoded['attention_mask'])
        
    return input_ids, attention_mask

# Cat lists to tensors for TensorDataset
def get_tensors(input_ids, attention_mask):
    
    input_ids = torch.cat(input_ids, dim=0)
    attention_mask = torch.cat(attention_mask, dim=0)
    
    return input_ids, attention_mask

# Import tools for Dataloader
from torch.utils.data import TensorDataset,DataLoader,RandomSampler,SequentialSampler

# Convert df to DataLoader
def get_dataloader(df, batch_size=32):
    temp_ids, temp_masks = get_encoded_dict(df)
    
    # Convert to tensors
    temp_ids, temp_masks = get_tensors(temp_ids, temp_masks)
    temp_labels = torch.tensor(df['label'].values)
    
    # Generate dataset
    temp_dataset = TensorDataset(temp_ids,
                                 temp_masks,
                                 temp_labels)
    
    # Generate dataloader
    temp_dataloader = DataLoader(temp_dataset,
                                 batch_size=batch_size,
                                 sampler=RandomSampler(temp_dataset))
    
    return temp_dataloader

# Get dataloader for all dataframes
train_dataloader = get_dataloader(df_train)
val_dataloader = get_dataloader(df_val)
test_dataloader = get_dataloader(df_test)

"""### Import BERT"""

from transformers import BertForSequenceClassification, AdamW

model = BertForSequenceClassification.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", num_labels=len(label_dict),
                                                      output_attentions=False,
                                                      output_hidden_states=False)



# Tell mode to use CUDA
model.cuda()

# Configuring optimizer
optimizer = AdamW(model.parameters(),
                  lr = 3e-5)

# Configuring scheduler
from transformers import get_linear_schedule_with_warmup

# Total steps: number of batchers * epochs
total_steps = len(train_dataloader) * epochs

# Set up the scheduler
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=10,
                                            num_training_steps=total_steps)

"""### Training"""

# Import materics
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import time

import numpy as np
# Reset history lists
training_stats = []

for epoch_i in range(epochs):
    
    # timer_start
    epoch_t0 = time.time()
    
    print('【EPOCH: {}/ {}】'.format(epoch_i+1, epochs))
    print('Trainig Phase')
    
    # Set training mode
    model.train()
    
    # Reset training loss
    total_training_loss = 0
    
    # Batch and forward
    for batch in train_dataloader:
        b_input_ids = batch[0].to(device)
        b_masks = batch[1].to(device)
        b_labels = batch[2].to(device)
    
        # Reset gradients before 
        model.zero_grad()
        
        # Forward pass
        res = model(b_input_ids,
                    token_type_ids=None,
                    attention_mask=b_masks,
                    return_dict=True)
        
        loss = res.loss
        logits = res.logits
        
        # sumup training loss
        total_training_loss += loss.item()
        
        # backpropagation
        loss.backward()
        
        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # update optimizer and scheduler
        optimizer.step()
        scheduler.step()
        
    # averrage loss
    avg_train_loss = total_training_loss/len(train_dataloader)
    print("  Average training loss: {0:.4f}".format(avg_train_loss))
    
    
    # validation
    print('Validation Phase')
    
    # Reset validation loss
    total_val_loss = 0
    
    # Set up lists
    ls_val_logits = []
    ls_val_labels = []

    # Get batchs from val_dataloader
    for batch in val_dataloader:
        b_input_ids = batch[0].to(device)
        b_masks = batch[1].to(device)
        b_labels = batch[2].to(device)

        # No need to calculate gradients
        with torch.no_grad():

            res = model(b_input_ids,
                        token_type_ids=None,
                        attention_mask=b_masks,
                        labels=b_labels,
                        return_dict=True)

        val_loss = res.loss
        val_logits = res.logits
        total_val_loss += val_loss.item()

        # Convert logitis to numpy format
        val_logits = np.argmax(val_logits.cpu().detach().numpy(), axis=1)
        val_labels = b_labels.cpu().detach().numpy()

        # Append data to the lists
        for logit in val_logits:
            ls_val_logits.append(logit)

        for label in val_labels:
            ls_val_labels.append(label)
    
    # Get accuracy score and val_loss
    acc = accuracy_score(ls_val_logits, ls_val_labels)
    avg_val_loss = total_val_loss/len(val_dataloader)
    
    # Print out validation performance
    print('  Average validation loss: {:.4f}'.format(avg_val_loss))
    print('  Validation accruacy: {:.4f}'.format(acc))
    
    
    # timer_end
    epoch_time_spent = time.time() - epoch_t0
    print('  Time spent on the epoch: {:.2f}'.format(epoch_time_spent))
    print('\n')
    
    # Recording training stats
    training_stats.append(
        {
            'Epoch': epoch_i+1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'val_acc': acc,
            'time_spent': epoch_time_spent
        })

"""### Training performance"""

train_loss = []
val_loss = []
val_acc = []
time_spent = []

for i in range(len(training_stats)):
    train_loss.append(training_stats[i]['train_loss'])
    val_loss.append(training_stats[i]['val_loss'])
    val_acc.append(training_stats[i]['val_acc'])
    time_spent.append(training_stats[i]['time_spent'])

# Print time spent
print('Time spent on training {} epochs: {:.0f}'.format(epochs, np.sum(time_spent)/60) + ' minutes')

# Plotting
import matplotlib.pyplot as plt

plt.figure(figsize=(10,3))

plt.subplot(1,2,1)
plt.plot(train_loss)
plt.plot(val_loss)
plt.title('loss')

plt.subplot(1,2,2)
plt.plot(val_acc, color='red', linestyle='--')
plt.title('accuracy')

torch.save(model.state_dict(), model_path)

"""### Perform on the test data"""

# Prepare df_test for prediction
t_input_ids, t_attention_mask = get_encoded_dict(df_test)
t_input_ids, t_attention_mask = get_tensors(t_input_ids, t_attention_mask)

# Prepare dataset and dataloader
test_dataset = TensorDataset(t_input_ids, t_attention_mask)
test_dataloader = DataLoader(test_dataset,
                             batch_size=32,
                             sampler=SequentialSampler(test_dataset))

# Show dataloader length
print('Number of batches in the dataloader: {}'.format(len(test_dataloader)))

# Setup lists for predictions and labels
ls_test_pred = []

# Get batchs from test_dataloader
for batch in test_dataloader:
    b_input_ids = batch[0].to(device)
    b_masks = batch[1].to(device)
    
    with torch.no_grad():

        res = model(b_input_ids,
                    attention_mask=b_masks,
                    return_dict=True)

        test_logits = res.logits
        test_logits = np.argmax(test_logits.cpu().detach().numpy(), axis=1)
        
        for pred in test_logits:
            ls_test_pred.append(pred)

# Set up list of test labels
ls_test_labels = df_test['Label'].values
rev_subs = { v:k for k,v in label_dict.items()}
ls_test_pred = [rev_subs.get(item,item)  for item in ls_test_pred]
# Get accuracy score and val_loss
acc = accuracy_score(ls_test_pred, ls_test_labels)
print('Prediction accuracy: {:.4f}'.format(acc))

label_dict.keys()

# Confusion matrix
cm = confusion_matrix(ls_test_labels, ls_test_pred)
labels = label_dict.keys()

fig = plt.figure(figsize=(8,8))
ax= fig.add_subplot(1,1,1)
sns.heatmap(cm, annot=True, cmap="Greens",ax = ax,fmt='g'); #annot=True to annotate cells

# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
ax.set_title('Confusion Matrix');
ax.xaxis.set_ticklabels(labels); ax.yaxis.set_ticklabels(labels);
plt.setp(ax.get_yticklabels(), rotation=30, horizontalalignment='right')
plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
plt.show()
print(classification_report(list(ls_test_labels),list(ls_test_pred),labels=list(labels)))

"""Clinical Bert on augmented data"""

epochs = 20
max_len = 300

# Experiment_name
experiment_name = 'basline_entire_dataset'

# Naming: model
model_path = '/content/drive/My Drive/CBB-750/Final Project/model/' + experiment_name + '.pt'
print(model_path)

# Naming: training history
training_df_path = '/content/drive/My Drive/CBB-750/Final Project/history/' + experiment_name + '.csv'
print(training_df_path)

"""### GPU"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor

from typing import Tuple, List

import random
import math
import os
import time
import json
import numpy as np
from collections import Counter

# We'll set the random seeds for deterministic results.
#SEED = 1

#random.seed(SEED)
#torch.manual_seed(SEED)
#torch.backends.cudnn.enabled = False
#torch.backends.cudnn.deterministic = True



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Pytorch version is: ", torch.__version__)
print("You are using: ", DEVICE)

# mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

"""### Read Dataset"""

import pandas as pd

df = pd.read_csv('/content/drive/My Drive/CBB-750/Final Project/augmented_data.csv')
df_gpt = pd.read_csv('/content/drive/My Drive/CBB-750/Final Project/GPTMessageLabelCode.csv')
df_gpt['Quotation Content'] = df_gpt['Message']
df = pd.concat([df, df_gpt], ignore_index=True)
# Combine all dataframes into one dataframe

df_combined

#df = df[df['Label']!="SDM"]
#df_1 = df[df['Label']=="Info Giving"]
#from random import randint
#x = [randint(0, 158) for p in range(0, 98)]
#df = df.drop(x)

# import package and torch
!pip install --upgrade packaging --quiet
!pip install torch --quiet
import torch

assert torch.cuda.is_available()

# Tell torch to use GPU
device = torch.device("cuda")
print('Running GPU: {}'.format(torch.cuda.get_device_name()))

possible_labels = df.Label.unique()

label_dict = {}
for index, possible_label in enumerate(possible_labels):
    label_dict[possible_label] = index
label_dict

df['label'] = df.Label.replace(label_dict)
df['text'] = df['Quotation Content']

import seaborn as sns
sns.histplot(df['Label'])

"""### Data Split"""

from sklearn.model_selection import train_test_split

df_train, df_rem = train_test_split(df,
                                    test_size=0.3,
                                    random_state=1,
                                    stratify=df['label'])

df_val, df_test = train_test_split(df_rem, test_size=0.5, random_state=1,
                                   stratify=df_rem['label'])

for item in df_train, df_val, df_test:
    print('Shape: {}'.format(item.shape))

# Show df_train class distribution
sns.histplot(df_train['Label'])
sns.histplot(df_val['Label'])
sns.histplot(df_test['Label'])

"""### Tokenization"""

!pip install transformers --quiet

import transformers
from transformers import BertTokenizer

# Load BERT tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name,
                                          do_lower_case=True)

def get_encoded_dict(df):
    input_ids = []
    attention_mask = []

    for text in df['Quotation Content']:
        encoded = tokenizer.encode_plus(text,
                                        add_special_tokens=True,
                                        padding='max_length',
                                        return_attention_mask=True,
                                        max_length=max_len,
                                        return_tensors='pt',
                                        truncation=True)

        input_ids.append(encoded['input_ids'])
        attention_mask.append(encoded['attention_mask'])

    return input_ids, attention_mask

# Cat lists to tensors for TensorDataset
def get_tensors(input_ids, attention_mask):

    input_ids = torch.cat(input_ids, dim=0)
    attention_mask = torch.cat(attention_mask, dim=0)

    return input_ids, attention_mask

# Import tools for Dataloader
from torch.utils.data import TensorDataset,DataLoader,RandomSampler,SequentialSampler

# Convert df to DataLoader
def get_dataloader(df, batch_size=32):
    temp_ids, temp_masks = get_encoded_dict(df)

    # Convert to tensors
    temp_ids, temp_masks = get_tensors(temp_ids, temp_masks)
    temp_labels = torch.tensor(df['label'].values)

    # Generate dataset
    temp_dataset = TensorDataset(temp_ids,
                                 temp_masks,
                                 temp_labels)

    # Generate dataloader
    temp_dataloader = DataLoader(temp_dataset,
                                 batch_size=batch_size,
                                 sampler=RandomSampler(temp_dataset))

    return temp_dataloader

# Get dataloader for all dataframes
train_dataloader = get_dataloader(df_train)
val_dataloader = get_dataloader(df_val)
test_dataloader = get_dataloader(df_test)

"""### Import BERT"""

from transformers import BertForSequenceClassification, AdamW, BertForPreTraining

model = BertForSequenceClassification.from_pretrained(model_name,
                                                      num_labels=len(label_dict),
                                                      output_attentions=False,
                                                      output_hidden_states=False)

# Tell mode to use CUDA
model.cuda()

# Configuring optimizer
optimizer = AdamW(model.parameters(),
                  lr = 3e-5)

# Configuring scheduler
from transformers import get_linear_schedule_with_warmup

# Total steps: number of batchers * epochs
total_steps = len(train_dataloader) * epochs

# Set up the scheduler
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=10,
                                            num_training_steps=total_steps)

"""### Training"""

# Import materics
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import time

import numpy as np
# Reset history lists
training_stats = []

for epoch_i in range(epochs):

    # timer_start
    epoch_t0 = time.time()

    print('【EPOCH: {}/ {}】'.format(epoch_i+1, epochs))
    print('Trainig Phase')

    # Set training mode
    model.train()

    # Reset training loss
    total_training_loss = 0

    # Batch and forward
    for batch in train_dataloader:
        b_input_ids = batch[0].to(device)
        b_masks = batch[1].to(device)
        b_labels = batch[2].to(device)

        # Reset gradients before
        model.zero_grad()

        # Forward pass
        res = model(b_input_ids,
                    token_type_ids=None,
                    attention_mask=b_masks,
                    return_dict=True,
                    labels=b_labels)

        loss = res.loss
        logits = res.logits

        # sumup training loss
        total_training_loss += loss.item()

        # backpropagation
        loss.backward()

        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # update optimizer and scheduler
        optimizer.step()
        scheduler.step()

    # averrage loss
    avg_train_loss = total_training_loss/len(train_dataloader)
    print("  Average training loss: {0:.4f}".format(avg_train_loss))


    # validation
    print('Validation Phase')

    # Reset validation loss
    total_val_loss = 0

    # Set up lists
    ls_val_logits = []
    ls_val_labels = []

    # Get batchs from val_dataloader
    for batch in val_dataloader:
        b_input_ids = batch[0].to(device)
        b_masks = batch[1].to(device)
        b_labels = batch[2].to(device)

        with torch.no_grad():

            res = model(b_input_ids,
                        token_type_ids=None,
                        attention_mask=b_masks,
                        labels=b_labels,
                        return_dict=True)

        val_loss = res.loss
        val_logits = res.logits
        total_val_loss += val_loss.item()

        # Convert logitis to numpy format
        val_logits = np.argmax(val_logits.cpu().detach().numpy(), axis=1)
        val_labels = b_labels.cpu().detach().numpy()

        # Append data to the lists
        for logit in val_logits:
            ls_val_logits.append(logit)

        for label in val_labels:
            ls_val_labels.append(label)

    # Get accuracy score and val_loss
    acc = accuracy_score(ls_val_logits, ls_val_labels)
    avg_val_loss = total_val_loss/len(val_dataloader)

    # Print out validation performance
    print('  Average validation loss: {:.4f}'.format(avg_val_loss))
    print('  Validation accruacy: {:.4f}'.format(acc))


    # timer_end
    epoch_time_spent = time.time() - epoch_t0
    print('  Time spent on the epoch: {:.2f}'.format(epoch_time_spent))
    print('\n')

    # Recording training stats
    training_stats.append(
        {
            'Epoch': epoch_i+1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'val_acc': acc,
            'time_spent': epoch_time_spent
        })

"""### Training performance"""

train_loss = []
val_loss = []
val_acc = []
time_spent = []

for i in range(len(training_stats)):
    train_loss.append(training_stats[i]['train_loss'])
    val_loss.append(training_stats[i]['val_loss'])
    val_acc.append(training_stats[i]['val_acc'])
    time_spent.append(training_stats[i]['time_spent'])

# Print time spent
print('Time spent on training {} epochs: {:.0f}'.format(epochs, np.sum(time_spent)/60) + ' minutes')

# Plotting
import matplotlib.pyplot as plt

plt.figure(figsize=(10,3))

plt.subplot(1,2,1)
plt.plot(train_loss)
plt.plot(val_loss)
plt.title('loss')

plt.subplot(1,2,2)
plt.plot(val_acc, color='red', linestyle='--')
plt.ylim(0.25,0.8)
plt.title('accuracy')

torch.save(model.state_dict(), model_path)

"""### Perform on the test data"""

# Prepare df_test for prediction
t_input_ids, t_attention_mask = get_encoded_dict(df_test)
t_input_ids, t_attention_mask = get_tensors(t_input_ids, t_attention_mask)

# Prepare dataset and dataloader
test_dataset = TensorDataset(t_input_ids, t_attention_mask)
test_dataloader = DataLoader(test_dataset,
                             batch_size=32,
                             sampler=SequentialSampler(test_dataset))

# Show dataloader length
print('Number of batches in the dataloader: {}'.format(len(test_dataloader)))

# Setup lists for predictions and labels
ls_test_pred = []

# Get batchs from test_dataloader
for batch in test_dataloader:
    b_input_ids = batch[0].to(device)
    b_masks = batch[1].to(device)

    with torch.no_grad():

        res = model(b_input_ids,
                    attention_mask=b_masks,
                    return_dict=True)

        test_logits = res.logits
        test_logits = np.argmax(test_logits.cpu().detach().numpy(), axis=1)

        for pred in test_logits:
            ls_test_pred.append(pred)

# Set up list of test labels
ls_test_labels = df_test['Label'].values
rev_subs = { v:k for k,v in label_dict.items()}
ls_test_pred = [rev_subs.get(item,item)  for item in ls_test_pred]
# Get accuracy score and val_loss
acc = accuracy_score(ls_test_pred, ls_test_labels)
print('Prediction accuracy: {:.4f}'.format(acc))

# Confusion matrix
cm = confusion_matrix(ls_test_labels, ls_test_pred)
labels = ["Emotion","Info Giving","Info Seeking","Partnership","SDM"]

fig = plt.figure(figsize=(8,8))
ax= fig.add_subplot(1,1,1)
sns.heatmap(cm, annot=True, cmap="Greens",ax = ax,fmt='g'); #annot=True to annotate cells

# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
ax.set_title('Confusion Matrix');
ax.xaxis.set_ticklabels(labels); ax.yaxis.set_ticklabels(labels);
plt.setp(ax.get_yticklabels(), rotation=30, horizontalalignment='right')
plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
plt.show()

print(classification_report(list(ls_test_labels),list(ls_test_pred),labels=list(labels)))

labels
