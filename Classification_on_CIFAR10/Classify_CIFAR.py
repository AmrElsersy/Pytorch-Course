import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import cv2
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import PIL

from ConvNet import *

class CIFAR10_Classifier:

    def __init__(self):

        # Hyperparameters
        self.batch_size = 4
        self.epochs = 20
        self.learning_rate = 0.001

        # Model & Optimizer & Loss
        self.net = ConvNet()
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=self.learning_rate)
        self.loss = nn.CrossEntropyLoss()

        self.read_dataset()
        self.weights_path = "./weights.pth"
        self.accuracy = 0.0

        # Try to load the weights if exist don't train the model
        try:
            weights = torch.load(self.weights_path)
            self.net.load_state_dict(weights)
            print("Weights loaded")
            self.test()
        except:
            self.train()


        self.classes = ['plane','car','bird','cat','deer','dog','frog','horse','ship','truck']

    def read_dataset(self):

        # Vision Datasets in torchvision came with images in range [0, 1] 
        # convert it to range [-1, 1] for a better convolutional training 
        # Resize it to 32 * 32 to match the input of the first layer
        self.transform = transforms.Compose([
            transforms.Resize(32,32),
            transforms.ToTensor(), 
            transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)) # for each channel to convert it to [-1,1] range
        ])

        self.train_dataset = torchvision.datasets.CIFAR10(root="./data", train=True, transform=self.transform, download=True)
        self.test_dataset = torchvision.datasets.CIFAR10(root="./data", train=False, transform=self.transform, download=False)

        self.train_loader = DataLoader(dataset=self.train_dataset, batch_size=self.batch_size)
        self.test_loader = DataLoader(dataset=self.test_dataset, batch_size=self.batch_size)
        
        print(f"Train dataset size= {len(self.train_dataset)} & image.shape = {self.train_dataset[0][0].shape}")

    def show_sample(self, index):

        image, label = self.train_dataset[index]
        image = image.view(32,32,3)
        print(image.shape)

        # # convert to Numpy image to show with openCV
        # np_image = image.numpy()        
        # print(np_image.shape)
        # cv2.imshow("ray2", np_image )
        # cv2.waitKey(0)

    def train(self):

        total_samples = len(self.train_dataset)
        batches_per_epoch = total_samples/self.batch_size

        for epoch in range(self.epochs):
            for i, (images, labels) in enumerate(self.train_loader):
                # forward
                labels_predicted = self.net(images)
                # loss
                loss = self.loss(labels_predicted, labels)
                # back propagation
                loss.backward()
                # update
                self.optimizer.step()
                self.optimizer.zero_grad()

                if i % (self.batch_size-1) == 0:
                    print(i+1,"/",batches_per_epoch,"batch .... ", epoch+1,"/",self.epochs, "epoch .... loss = ", loss.item())
                    # each predicted is (batch_size, 10) ..... labels is (batch_size, 1) represents the classification answer
                    # print(labels_predicted.shape, labels.shape)

        torch.save(self.net.state_dict(), self.weights_path)


    def test(self):

        n_correct = 0
        n_samples = len(self.test_dataset)

        with torch.no_grad():
            for images, labels in self.test_loader:

                predicted_labels = self.net(images)
                # get index instead of ( values list for each image ) .. so we will have just the output(index of classes) of the image 
                values, indexes = torch.max(predicted_labels, 1)
                n_correct += (indexes == labels).sum().item()

            self.accuracy = n_correct / n_samples
            print(f"Accuracy = {self.accuracy}")


    def get_accuracy(self):
        return self.accuracy

    def classify(self, img):

        with torch.no_grad():
            # transform the shape & to tensor & norimalize
            # self.transform(img)
            # label = self.net(img)
            # _, index = torch.max(label, 1)
            # print(self.classes[index])
            pass
        


# image =  Image.open("dog.jpeg")
# image = image.resize((32,32))
# pixels = np.asarray(image)
# pixels = pixels / 255
# image = Image.fromarray(pixels.astype('float'), 'RGB')
# image = transforms.ToTensor()(image)
# print(image.shape, type(image))

classifier = CIFAR10_Classifier()


