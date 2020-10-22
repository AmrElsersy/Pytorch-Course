import torch, torchvision
from torch.utils.data import DataLoader, Dataset
import numpy as np
import math
import pandas 


from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder, StandardScaler 
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer, SimpleImputer

class TitanicDataset(Dataset):
    def __init__(self):
        # loading data ............... delimiter : text spliter for csv ...... skip first row
        self.dataset = pandas.read_csv("dummy_titanic.csv", sep=",").to_numpy()

        self.n_samples = self.dataset.shape[0]

        self.x = torch.tensor(self.dataset[: , 1:].astype(np.float32))
        self.y = torch.tensor(self.dataset[: ,0].astype(np.float32))

        self.y = self.y.view(self.y.shape[0],1)

    def __getitem__(self, i):
        return self.x[i], self.y[i]
    
    def __len__(self):
        return self.n_samples


dataset = TitanicDataset()

batch_size = 4
epochs = 3
iterations_per_epoch = len(dataset) / batch_size

# data loader for easy axis to batches
data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

for epoch in range(epochs):

    for i, (input, output) in enumerate(data_loader):
        if i % 5 == 0:
            print(i," ... iter=", i ,"/",iterations_per_epoch, "... epoch=", epoch+1, "/", epochs)



# ==================== Pytorch Transform =======================


# Pytorch has many transform functions
transform = torchvision.transforms.ToTensor() # return ref to a function

numpyArr = np.array( [ [2,3,4], [3,1,7] ] )
tensorArr = transform(numpyArr)
print(tensorArr)
