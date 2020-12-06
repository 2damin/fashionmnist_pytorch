import torch 
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import matplotlib.pylab as plt
import numpy as np

torch.manual_seed(0)
def save_data(data_sample, data_name):
    plt.imsave(data_name,data_sample[0].numpy().reshape(IMAGE_SIZE, IMAGE_SIZE), cmap='gray')
    plt.title('y = '+ str(data_sample[1].item()))

class CNN_batch(nn.Module):
    
    # Contructor
    def __init__(self, out_1=6, out_2=16,number_of_classes=10):
        super(CNN_batch, self).__init__()
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=out_1, kernel_size=5, padding=0, stride=1)
        torch.nn.init.kaiming_uniform_(self.cnn1.weight, nonlinearity='relu')
        self.conv1_bn = nn.BatchNorm2d(out_1)
        self.avgpool1=nn.AvgPool2d(kernel_size=2)
        
        self.cnn2 = nn.Conv2d(in_channels=out_1, out_channels=out_2, kernel_size=5, stride=1, padding=0)
        torch.nn.init.kaiming_uniform_(self.cnn2.weight, nonlinearity='relu')
        self.conv2_bn = nn.BatchNorm2d(out_2)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2)

        self.fc1 = nn.Linear(out_2 * 5 * 5, 120)
        torch.nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        self.bn_fc1 = nn.BatchNorm1d(120)
        
        self.fc2 = nn.Linear(120, 84)
        torch.nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')
        self.bn_fc2 = nn.BatchNorm1d(84)

        self.fc3 = nn.Linear(84, number_of_classes)
        torch.nn.init.kaiming_uniform_(self.fc3.weight, nonlinearity='relu')
        self.bn_fc3 = nn.BatchNorm1d(number_of_classes)
    
    # Prediction
    def forward(self, x):
        x = self.cnn1(x)
        x=self.conv1_bn(x)
        x = torch.relu(x)
        x = self.avgpool1(x)
        x = self.cnn2(x)
        x=self.conv2_bn(x)
        x = torch.relu(x)
        x = self.avgpool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x=self.bn_fc1(x)
        x = self.fc2(x)
        x=self.bn_fc2(x)
        x = self.fc3(x)
        x=self.bn_fc3(x)
        return x

def train_model(model,train_loader,validation_loader,optimizer,n_epochs=4, device = torch.device("cuda:0")):
    
    #global variable 
    N_test=len(validation_dataset)
    accuracy_list=[]
    loss_list=[]
    accuracy = 0
    for epoch in range(n_epochs):
        for x, y in train_loader:
            model.train()
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            z = model(x)
            loss = criterion(z, y)
            loss.backward()
            optimizer.step()
            loss_list.append(loss.data)
            accuracy_list.append(accuracy)
        
        #remove last accuracy element
        accuracy_list.pop()

        correct=0
        #perform a prediction on the validation  data  
        for x_test, y_test in validation_loader:
            x_test,y_test = x_test.to(device), y_test.to(device)
            model.eval()
            z = model(x_test)
            _, yhat = torch.max(z.data, 1)
            correct += (yhat == y_test).sum().item()
        accuracy = correct / N_test
        accuracy_list.append(accuracy)
     
    return accuracy_list, loss_list


IMAGE_SIZE = 32

composed = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor()])

train_dataset = dsets.MNIST(root='./data', train=True, download=True, transform=composed)

validation_dataset = dsets.MNIST(root='./data', train=False, download=True, transform=composed)

save_data(validation_dataset[0], "validation_0.jpg")
save_data(validation_dataset[1], "validation_1.jpg")
save_data(validation_dataset[2], "validation_2.jpg")

print(torch.cuda.is_available())
device = torch.device("cuda:0")

model = CNN_batch()
model.to(device)

criterion = nn.CrossEntropyLoss()
learning_rate = 0.01
optimizer = torch.optim.Adam(model.parameters(), learning_rate)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=1000)
valid_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=5000)

accuracy_list, loss_list = train_model(model,train_loader, valid_loader,optimizer, 20, device)

plt.plot(loss_list, 'b',label='cost')
plt.plot(accuracy_list, 'r',label='accuracy')
plt.ylabel('cost')
plt.ylabel('accuracy')
plt.xlabel('iteration')
plt.title("cost&accuracy")
plt.legend()
plt.savefig("cost_accuracy.jpg")
plt.clf()
