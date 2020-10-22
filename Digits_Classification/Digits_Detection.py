import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

# ========== Hyperparameters ========
batch_size = 100
epochs = 2
learning_rate = 0.01

input_size = 784 # 28*28
output_size = 10 # 10 classes of numbers
hidden_size1 = 90
hidden_size2 = 45

saved_weights_path = "./weights.pth"

# =========== Dataset ===============
MNIST_Transform = transforms.Compose( [transforms.Resize((28,28)), transforms.ToTensor()] )

train_dataset = torchvision.datasets.MNIST(root= "./MNIST", train=True, transform=MNIST_Transform, download=True)
test_dataset  = torchvision.datasets.MNIST(root= "./MNIST", train=False, transform=MNIST_Transform, download=False)

# print(len(train_dataset), "Samples")
# image, label = train_dataset[50000]
# print(image.shape, "  & Class = ", label)

train_loader = DataLoader(dataset= train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(dataset= test_dataset, batch_size=batch_size, shuffle=False)

# show some images of the dataset
# for i in range(8):
#     image_to_plot, _ = train_dataset[i]
#     plt.subplot(2,4, i+1)
#     plt.imshow(image_to_plot[0], cmap="gray")
# plt.show()

# ========== Nerual Network ================
class DigitModel_DNN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(DigitModel_DNN, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.l2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.l3 = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        # flatten
        x = x.view(-1, 28*28)

        out = self.l1(x) 
        out = self.relu1(out)
        out = self.l2(out)
        out = self.relu2(out)
        out = self.l3(out)

        return out


# ========= Model & Loss & Optimizer ========
model = DigitModel_DNN(input_size, hidden_size1, hidden_size2, output_size)
criterion = nn.CrossEntropyLoss()
Optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

total_samples = len(train_dataset)
batches_per_epoch = total_samples/batch_size

# ============ Training ===============
def train_model():
    global epochs, train_loader, model, criterion, Optimizer, batches_per_epochs
    for epoch in range(epochs):
        for i, (images, labels) in enumerate(train_loader):
            # forward
            labels_predicted = model(images)
            # loss
            loss = criterion(labels_predicted, labels)
            # back propagation
            loss.backward()
            # update
            Optimizer.step()
            Optimizer.zero_grad()

            if i % (batch_size-1) == 0:
                print(i+1,"/",batches_per_epoch,"batch .... ", epoch+1,"/",epochs, "epoch .... loss = ", loss.item())


    torch.save(model.state_dict(), saved_weights_path)
# ============= Testing ==============

def detect_digit(index):
    global test_dataset, model
    with torch.no_grad():
        test_image, test_label = test_dataset[index]
        labels = model(test_image)
        values, indexes = torch.max(labels, 1)
        print("Image index ",index , " digit = ", indexes.item())
        plt.imshow(test_image[0], cmap="gray")
        plt.show()


def test_model():
    global test_loader, model
    with torch.no_grad():
        correct_n = 0
        for i, (images, labels) in enumerate(test_loader):

            outputs = model(images)
            values, indexes = torch.max(outputs, 1)
            correct_n += (indexes == labels).sum().item()

        total_test_samples = len(test_dataset)
        accuracy = (correct_n/ total_test_samples )*100.0
        print(f"Accuracy = {correct_n}/{total_test_samples} = {accuracy}")
        return accuracy
            

try:
    weights = torch.load(saved_weights_path)
    model.load_state_dict(weights)
    # test_model()
    print(len(test_dataset))
    detect_digit(44)
except:
    train_model()
    test_model()
