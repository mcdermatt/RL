import torch
import torchvision
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

start = time.time()

# torch.set_default_tensor_type('torch.cuda.FloatTensor')

if torch.cuda.is_available():
    device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
    torch.set_default_tensor_type('torch.cuda.FloatTensor') 
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    torch.set_default_tensor_type('torch.FloatTensor')
    print("Running on the CPU")

# device = torch.device("cpu")
# print("Running on the CPU lol")

# #download datasets locally
train = datasets.MNIST("", train = True, download = False,
                      transform = transforms.Compose([transforms.ToTensor()]))
test = datasets.MNIST("", train = False, download = False,
                      transform = transforms.Compose([transforms.ToTensor()]))

#batches - don't want to give the algo all the data at once or it will not work as well on new data
#   usually best batch size is between 8 and 64
trainset = torch.utils.data.DataLoader(train, batch_size = 512, shuffle = True) #GPU does better on higher batch size in this situation (not that its really)
print(type(trainset))
print(trainset.shape)
testset = torch.utils.data.DataLoader(test, batch_size = 512, shuffle = True)

# Balancing is SUPER important- if 60% of our training data is a "3" then a set of weights that always guesses 3 will reach a local max and training will fail
# total = 0
# counter_dict = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0}

# for data in trainset:
#     Xs, ys = data
#     for y in ys:
#         counter_dict[int(y)] += 1
#         total += 1
        
# print(counter_dict)
# for i in counter_dict:
    # print(f"{i}: {counter_dict[i]/total*100}")

class Net(nn.Module): #create class Net and inherit from nn.Module
    def __init__(self):
        super().__init__() #need to run this because the init func of nn.Module is not run from inherit
       
        #Linear is a simple flat fuly connected
        self.fc1 = nn.Linear(784, 64) #when images are flattened they are 28*28 = 784 long
        self.fc2 = nn.Linear(64, 64)  #arbitrarily choosing 64 nodes for hidden layers
        self.fc3 = nn.Linear(64, 64) 
        self.fc4 = nn.Linear(64, 10)  #output layer is size 10 for digits 0-9
        
    def forward(self, x):
        #F.relu is rectified linear activation func
        #   activation func is sigmoid- keeps output from exploding
        #   attempt to model whether neuron is or is not firing
        x = F.relu(self.fc1(x)) 
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        #for output we only want one neuron to be fully fired
        x = self.fc4(x)
        
        return F.log_softmax(x, dim = 1) #since I am working with flat linear funcs this dim will always = 1        
net = Net()
net = net.to(device)


# net.cuda() #has issues with nll_loss() method
# print(net)

#loss is a measure of how wrong is our model .loss should trend downward over time

#Optimizer adjusts weights bit by bit over time (according to learning rate) to lower loss
optimizer = optim.Adam(net.parameters(), lr = 0.001) #net.parmeters controls what stuff in net() is adjusted (default is everything)
#decaying learning rate -> lr decreases over time to prevent overshooting
EPOCHS = 3 #full passes through dataset
for epoch in range(EPOCHS):
    for data in trainset:
        #data is a batch of featuresets and labels
        X, y = data
        X, y = X.to(device), y.to(device) #send to device
        net.zero_grad()
        
        output = net(X.view(-1,28*28))
        loss = F.nll_loss(output, y) #usually use mean^2 error if data is onehot vector
        
        #magic backpropogation algorithm
        loss.backward()
        
        #adjusts weights
        optimizer.step()
        
    print(f"Epoch: {epoch}. Loss: {loss}")

correct = 0
total = 0

with torch.no_grad():
    #net.train() and net.eval() are depreciated ways of changing the mode
    for data in testset:
        X, y = data
        X, y = X.to(device), y.to(device) #send to device
        output = net(X.view(-1, 784))
        
        for idx, i in enumerate(output):
            if torch.argmax(i) == y[idx]:
                correct += 1
                
            total += 1
print("Accuracy: ", round(correct/total, 3))
finish = time.time()
print("training took ", finish-start, " seconds" )