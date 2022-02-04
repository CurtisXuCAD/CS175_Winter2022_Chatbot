import json
from nltk_utils import *
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

with open('intents.json','r') as file:
     intents = json.load(file)

all_words = []
tags = []
xy = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w) # use extend instead of append because we dont want a array in a array
        xy.append((w,tag))

#ignore_words = 
all_words = [stem(word) for word in all_words if word not in ['?','.','!',',']]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

X_train = []
Y_train = []

for pattern_sentence, tag in xy:
    bow = bag_of_words(pattern_sentence, all_words)     #get each line of bow
    X_train.append(bow)       #append to X_train

    label = tags.index(tag)   # make label numbers
    Y_train.append(label)     #cross entropy loss

X_train = np.array(X_train)
Y_train = np.array(Y_train)

class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = Y_train 
    
    #dataset[idx]
    def __getitem__(self, index):
        return self.x_data[idx], self.y_data[idx]

    def __len__(self):
        return self.n_samples

#Hyperparameters
batch_size = 8


dataset = ChatDataset()
train_loader = DataLoader(dataset = dataset, batch_size = batch_size, shuffle = True, num_workers = 2)