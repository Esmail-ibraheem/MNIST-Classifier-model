from turtle import forward
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch
import torch.nn as nn 
import torch.optim as optim 
import matplotlib.pyplot as plt

training_data = datasets.MNIST(root = "data", 
                               train = True, 
                               transform= ToTensor, 
                               download=True)

testing_data = datasets.MNIST(root = "data", 
                              train = True, 
                              transform= ToTensor, 
                              download=True)

loaders = {'train': DataLoader(training_data, 
                               batch_size=100, 
                               shuffle= True, 
                               num_workers=1),

           'test': DataLoader(training_data, 
                              batch_size=100, 
                              shuffle= True, 
                              num_workers=1),}


class Convultional_Neural_Network(nn.Module):
    def __init__(self):
        super(Convultional_Neural_Network, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv_dropout = nn.Dropout2d()
        self.fully_connected_layer = nn.Linear(320, 50)
        self.fully_connected_layer2 = nn.Linear(50, 10)

    def forward(self, x):
        x = nn.functional.relu(nn.functional.max_pool2d(self.conv1(x), 2))
        x = nn.functional.relu(nn.functional.max_pool2d(self.conv_dropout(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = nn.functional.relu(self.fully_connected_layer(x))
        x = nn.functional.dropout(x , training= self.training)
        x = self.fully_connected_layer2(x) 
        return nn.functional.softmax(x)

device = ('cuda' if torch.cuda.is_available() else 'cpu')
model = Convultional_Neural_Network().to(device)
optimizer = optim.Adam(model.parameters(), lr = 0.001)
loss_function = nn.CrossEntropyLoss()

def train(epoch):
    model.train()
    for batch, (data, target) in enumerate(loaders['train']):
        data, targets = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()
        if batch%20==0:
            print(f'Train Epoch: {epoch} [{batch * len(data)}/ {len(loaders["train"].dataset)} ({100. * batch / len(loaders["train"]):.0f} %)]\t{loss.item():.6}')

def test():
    model.eval()

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in enumerate(loaders['train']):
            data, targets = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_function(output, target).item()
            prediction = output.argmax(dim = 1, keepdim = True)
            correct+=prediction.eq(target.view_as(prediction)).sum().item()
    test_loss/= len(loaders['train'].dataset)
    print(f'\n Test set: Average loss: {test_loss:.4f}, Occuracy{correct}/ {len(loaders["train"].dataset)} ({100. * correct/len(loaders["train"].dataset):.0f} %\n')

for epoch in range(1,11):
    train(epoch)
    test()




model.eval()

for i in range(len(testing_data)):
    data, target = testing_data[i]
    data = data.unsqueeze(0).to(device)
    output = model(data)
    predicted = output.argmax(dim =  1, keepdim = True).item()
    print(f"Preditced Value is : {predicted}")
    image = data.squeeze(0).squeeze(0).cpu().numpy()

    plt.imshow(image, cmap = 'gray')
    plt.show()
