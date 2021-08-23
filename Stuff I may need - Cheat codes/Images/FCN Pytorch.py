#!/usr/bin/env python
# coding: utf-8

# In[7]:


import os
import pandas as pd 
import matplotlib.pyplot as plt 
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.model_selection import train_test_split
import os
from PIL import Image


# In[8]:


pregaT = list(os.listdir('Dataset/Prega'))
prodT = list(os.listdir('Dataset/ProdutoConforme'))
sujiT = list(os.listdir('Dataset/Sujidade'))

pregaV = list(os.listdir('Teste/Prega'))
prodV = list(os.listdir('Teste/ProdutoConforme'))
sujiV = list(os.listdir('Teste/Sujidade'))


# In[9]:


pregaV.remove('.ipynb_checkpoints')


# In[10]:


pathtrain = pregaT + prodT + sujiT
dftrain = pd.DataFrame(pathtrain, columns=['image'])

prefix = []
label = []

for i in dftrain['image']:
    if i.startswith('Prega'):
        prefix.append('Prega/' + i)
        label.append('ProdutoNaoConforme')
    elif i.startswith('Produto'):
        prefix.append('ProdutoConforme/' + i)
        label.append('ProdutoConforme')
    elif i.startswith('Sujidade'):
        prefix.append('Sujidade/' + i)
        label.append('ProdutoNaoConforme')
    
dftrain = pd.DataFrame()


# In[11]:


dftrain['image'] =  prefix
dftrain['label'] =  label

repl={}
classes=list(dftrain['label'].unique())
for i in range(len(classes)):
    repl[classes[i]]=i
for i in range(len(dftrain)):
    dftrain['label'][i]=repl[dftrain['label'][i]]


# In[12]:


pathval = pregaV + prodV + sujiV
dfval = pd.DataFrame(pathval, columns=['image'])

label2 = []
prefix2 = []

for i in dfval['image']:
    if 'Prega' in i:
        prefix2.append('Prega/' + i)        
        label2.append(0)
    elif 'Produto' in i:
        prefix2.append('ProdutoConforme/' + i)        
        label2.append(1)
    elif 'Sujidade' in i:
        prefix2.append('Sujidade/' + i)       
        label2.append(0)

dfval =pd.DataFrame()


# In[13]:


dfval['label'] =  label2
dfval['image'] = prefix2


# In[14]:


dftrain.to_csv('train.csv', index=False)
dfval.to_csv('val.csv', index=False)


# In[15]:


class dataset(Dataset):
    def __init__(self, root_dir, annotation_file, transform=None):
        self.root_dir = root_dir
        self.annotations = pd.read_csv(annotation_file)
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_id = self.annotations.iloc[index, 0]
        img = Image.open(self.root_dir+ img_id).convert("RGB")
        y_label = torch.tensor(float(self.annotations.iloc[index, 1]))

        if self.transform is not None:
            img = self.transform(img)

        return (img, y_label)
    
    
transform = transforms.Compose(
    [transforms.Resize((1000,1000)),transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 1

trainset = dataset(root_dir='./Dataset/', annotation_file='train.csv', transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = dataset(root_dir='./Teste/', annotation_file='val.csv', transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)


# In[16]:


class Conv1(torch.nn.Module):
    def __init__(self):
        
        super().__init__()
        self.weight1 = nn.Parameter(torch.randn(3, 20, 20))
        self.weight2 = nn.Parameter(torch.randn(3, 20, 20))

    def forward(self, x):
        
        batch_size = 1
        channels = 3
        image = x # input image

        kh, kw = 20, 20 # kernel size
        dh, dw = 20, 20 # stride

        # Create conv
        
        inputs=x
        patches = inputs.unfold(2, kh, dh).unfold(3, kw, dw)
        #print(patches.shape)
        patches = patches.contiguous().view(batch_size, channels, -1, kh, kw)
        #print(patches.shape)
        nb_windows = patches.size(2)

       
        patches = patches.permute(0, 2, 1, 3, 4)
        #print(patches.shape)
        patches = patches.view(-1, channels, kh, kw)
        #print(patches.shape)

        conv = nn.Conv2d(channels, batch_size, (kh, kw), stride=(dh, dw), bias=False)
        patches = patches * torch.mul(self.weight1, self.weight2)
        patches = patches.sum(1)
        patches=patches.unsqueeze(0)
        return patches


# In[17]:


import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 =Conv1()
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(250000, 20)
        

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.fc1(x)
        return x


net = Net()


# In[18]:


import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


# In[20]:


losses=[]
for epoch in range(6):  

    running_loss = 0.0
    i=0
    for inputs, labels in trainloader:
        
        optimizer.zero_grad()
        outputs = net(inputs.float())
        loss = criterion(outputs,labels.long())
        loss.backward()
        optimizer.step()        
        running_loss += loss.item()
        i+=1
        if i % 200 == 0:    
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 200))
            running_loss = 0.0
#     with torch.no_grad():
#         val_loss=0
#         for val_inputs,val_labels in testloader:
#             val_outputs = net(val_inputs.float())
#             loss = criterion(val_outputs,val_labels.long())     
#             val_loss += loss.item()
#         losses.append(val_loss)
#         print("Validation loss:",val_loss)

print('Finished Training')


# In[21]:


plt.plot(losses)


# In[22]:


img = Image.open('Dataset/ProdutoConforme/ProdutoConforme1.jpg').convert("RGB")
img = transform(img)


# In[23]:


outputs = net(img)
_, predicted = torch.max(outputs, 1)

predicted


# In[25]:


testeprega = list(os.listdir('Teste/ProdutoConforme'))
for i in testeprega:
    img = Image.open('Teste/ProdutoConforme/' + i).convert("RGB")
    img = transform(img)
    outputs = net(img)
    _, predicted = torch.max(outputs, 1)
    print(list(repl.keys())[list(repl.values()).index(predicted)]) 


# In[ ]:


torch.save(net, "./demo1.pth")


# In[ ]:


img = Image.open('Dataset/Prega/Prega1.jpg')
img.show()


# In[ ]:


img.resize(1000x1000)


# In[ ]:


print(list(repl.keys())[list(repl.values()).index(predicted)]) 


# In[ ]:


predicted


# In[ ]:




