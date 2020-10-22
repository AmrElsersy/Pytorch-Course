import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ============ data preprocessing ======================
cancer_dataset = datasets.load_breast_cancer()
X_numpy, Y_numpy = cancer_dataset.data, cancer_dataset.target

# split dataset
x_train, x_test, y_train, y_test = train_test_split(X_numpy, Y_numpy, test_size=0.15, random_state=1234)

# Normalization .... mean 0 & variance 1 (Recomended for Logistic Regression)
standard_scaler = StandardScaler()
x_train = standard_scaler.fit_transform(x_train)
x_test = standard_scaler.transform(x_test)

# convert data from numpy to tensors
x_train = torch.from_numpy(x_train.astype(np.float32))
x_test  = torch.from_numpy(x_test.astype (np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test  = torch.from_numpy(y_test.astype (np.float32))

# transpose ... convert the vector to a 1-column vector
y_train = y_train.view(y_train.shape[0], 1)
y_test  = y_test.view (y_test.shape[0], 1)

samples, n_features_input = x_train.shape
n_features_output = 1

print("dataset samples = ",samples, " & features = ",n_features_input, " & output_n = ", n_features_output)

# ============ Model ==============
class LogisticsRegression(nn.Module):
    def __init__(self, n_features, output_size):
        super(LogisticsRegression, self).__init__()
        self.Linear = nn.Linear(n_features, output_size)

    def forward(self, x):
        # w * x + b
        z = self.Linear(x)
        return torch.sigmoid(z)

model = LogisticsRegression(n_features_input, n_features_output)

# ============ Loss and Optimizer ==============
learning_rate = 0.01
entropy_loss = nn.BCELoss()
Optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# ============ Training Loop ==============
epochs = 200

for epoch in range(epochs):

    y_predicted = model(x_train)

    loss = entropy_loss(y_predicted, y_train)

    loss.backward()

    Optimizer.step()
    Optimizer.zero_grad()

    # if epoch % 10 == 0:
    #     print("loss = ", loss.item())


# with no grade not to become a part of the graph
with torch.no_grad():
    
    samples_test = x_test.shape[0] # or y_test

    y_predicted = model(x_test)

    # y_predicted is a number from 0 to 1 ... convert it to 0 or 1 ... could be y_predicted.round()
    y_binary = y_predicted >= 0.7

    accuracy = (y_binary == y_test).sum() / float(samples_test)

    # print("Accuracy = ", accuracy)



samples_test = x_test.shape[0] # or y_test
y_predicted = model(x_test)

print(y_predicted.grad_fn)


