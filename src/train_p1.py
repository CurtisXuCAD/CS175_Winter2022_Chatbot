
import json
from nltk_utils import *
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from neural_net import NeuralNet

with open('save\\intents.json','r') as file:
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
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

#Hyperparameters
batch_size = 8
hidden_size = 8
output_size = len(tags)
input_size = len(X_train[0]) #or len(all_words)
learning_rate = 0.001
num_epochs = 1000
# print(input_size)
# print(len(all_words))
# print(output_size)

dataset = ChatDataset()
train_loader = DataLoader(dataset = dataset, batch_size = batch_size, shuffle = True, num_workers = 0)#num_workers cannot be 2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size, hidden_size, output_size).to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

for epoch in range(num_epochs):
    for (words,labels) in train_loader:
        words = words.to(device)
        labels = labels.to(device)
        labels = labels.long()
    
        #forward
        outputs = model(words)
        loss = criterion(outputs, labels)

        #backward and optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 100 == 0:
        print(f'epoch{epoch+1}/{num_epochs}, loss = {loss.item():.4f}')

print(f'final loss, loss= {loss.item():.4f}')

data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "output_size": output_size,
    "hidden_size" : hidden_size,
    "all_words": all_words,
    "tags": tags

}

FILE = "save\\data.pth"
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}')
