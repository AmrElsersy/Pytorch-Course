import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset

import time, copy

# Transfer Learning Mode
fine_tuning_mode = False

dataset_dir = "data/"
batch_size = 4

# =================================== Datasets =======================================

# standard normalization coffs (mean & std deviation) for all models in Pytorch
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
min_size_input = 224

# (3 x H x W), where H and W are expected to be at least 224
transform_train = transforms.Compose([
    # crop a 224 * 224 sub image randomly
    transforms.RandomResizedCrop(min_size_input),
    # flip the image (maybe will not)
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # normalize for used models
    transforms.Normalize(mean=mean, std=std)
])

transform_test = transforms.Compose([
    # resize just the height of the image .. for full size pass a tuple (width_resized, height_resized)
    transforms.Resize(256),
    transforms.RandomResizedCrop(min_size_input),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

train_dataset = torchvision.datasets.ImageFolder(root=dataset_dir+"train", transform=transform_train)
test_dataset = torchvision.datasets.ImageFolder(root=dataset_dir+"val", transform=transform_test)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)


classes = train_dataset.classes
print(classes)

# =================================== Transfer Learning Resnet18 =======================================

resnet18 = torchvision.models.resnet18(pretrained=True)

if not fine_tuning_mode:
    # if fixed trainable parameters except the last layer
    for parameter in resnet18.parameters():
        parameter.requires_grad = False

in_features_fc = resnet18.fc.in_features

# multi fully connected layers
resnet18.fc = nn.Sequential(
    nn.Linear(in_features_fc, 100),
    nn.ReLU(),
    nn.Linear(100, 2)
    )

print(resnet18)

# Optimizer & Loss
learning_rate = 0.01
epochs = 20

optimizer = None

if fine_tuning_mode:
    optimizer = torch.optim.SGD(resnet18.parameters(), lr=learning_rate)
else:
    optimizer = torch.optim.SGD(resnet18.fc.parameters(), lr=learning_rate)

criterion = nn.CrossEntropyLoss()
lr_schedular = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=6, gamma=0.1)

total_samples = len(train_dataset)
batches_per_epoch = total_samples/batch_size

# =================================== Training one epoch =======================================

def train(model, optimizer, lr_schedular, criterion, train_loader):

    global total_samples, batches_per_epoch

    # training mode .. to notify all layers
    # some layers perform in a different way if train or eval mode like BatchNorm or Dropout 
    model.train()

    for i, (images, labels) in enumerate(train_loader):
        
        predicted_labels = resnet18(images)

        loss = criterion(predicted_labels, labels)

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        print(i+1,"/",batches_per_epoch,"batch .... loss = ", loss.item())

    # decay the learning rate (make it smaller) @ each epoch
    # because at last epochs we need to slow down the moving on the loss curve for better accuracy
    lr_schedular.step()
    
    return model

# =================================== Model Eval =======================================

def eval(model, test_loader, test_dataset):

    # model eval mode
    # some layers perform in a different way if train or eval mode like BatchNorm or Dropout 
    model.eval()

    # number of correct classifaction 
    n_correct = 0

    # disable autograd so operations will not be tracked
    with torch.no_grad():

        for i, (images, labels) in enumerate(test_loader):        
            # output contains values for each class (without softmax)
            predicted_labels = model(images)
            # get index of the (highest output of the predicted labels) (index in the classes array whith contains only bees and ants (0 or 1) )
            _, indexes = torch.max(predicted_labels, 1)

            # correct : index = label , 0 = 0
            n_correct += (indexes == labels).sum().item()
            
        accuarcy = n_correct / len(test_dataset) 

        return accuarcy * 100

# =================================== Training-Eval loop =======================================

def train_eval(model, optimizer, lr_schedular, criterion, train_loader, test_loader, test_dataset, epochs):

    t1 = time.time()

    best_accuarcy = 0
    best_weights = model.state_dict

    # evaluate @ each epoch to get both best accuracy and weights
    # note we compare according to accuracy not loss
    for epoch in range(epochs):
        print("epoch=",epoch)

        model = train(model, optimizer, lr_schedular, criterion, train_loader)
        accuarcy = eval(model, test_loader, test_dataset)

        if accuarcy > best_accuarcy:
            best_accuarcy = accuarcy
            best_weights = copy.deepcopy(model.state_dict())


    t2 = time.time() - t1

    # set the model weights with the weights of best accuracy
    model.load_state_dict(best_weights)

    return model, best_accuarcy        

# path to load weights
weights_path = "./weights_many_fc.pth"

try:

    # if weights is not found will through exception
    weights = torch.load(weights_path)
    # set the model's weights
    resnet18.load_state_dict(weights)
    print("Weights Loaded")

    # Test
    accuracy = eval(model=resnet18, test_loader=test_loader, test_dataset=test_dataset)
    print("Accuracy = ", accuracy)

except:

    # Train
    resnet18, best_accuracy = train_eval(resnet18,optimizer,lr_schedular,criterion,train_loader,test_loader, test_dataset ,epochs)

    # save model's weights 
    torch.save(resnet18.state_dict(), weights_path)

    print(best_accuracy)




