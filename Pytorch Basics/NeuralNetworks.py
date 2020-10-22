import torch
import torch.nn as nn
import torchvision

# Manual Neural Network
x = torch.tensor([[2,2], [4,5] ,[10,3]], dtype=torch.float)
y = torch.tensor([[4,6], [8,15],[20,9]], dtype=torch.float)

# same samples
samples , input_features_size = x.shape
samples , output_features_size = y.shape
 
print(x.shape)
print(y.shape)

model = nn.Linear(input_features_size, output_features_size)

epochs = 5000
learning_rate = 0.003

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters() , lr=learning_rate)

for epoch in range(epochs):
    
    y_predicted = model(x)

    Loss = loss(y_predicted, y)

    Loss.backward(torch.rand(Loss.size()))

    optimizer.step()
    optimizer.zero_grad()

    if epoch % 100 == 0 :
        [w, b] = model.parameters()
        print("w = ",  w[0], " & b = ", b[0], "  & loss = ", Loss.item())

test_x = torch.tensor([3,4], dtype=torch.float)
test_predicted = model(test_x)

print (test_predicted)