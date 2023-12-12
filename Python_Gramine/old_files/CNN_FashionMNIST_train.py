
import torch

import time
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

class CNN(torch.nn.Module):

    def __init__(self):
        super().__init__()

        # define layers
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)

        self.fc1 = torch.nn.Linear(in_features=12*4*4, out_features=120)
        self.fc2 = torch.nn.Linear(in_features=120, out_features=60)
        self.out = torch.nn.Linear(in_features=60, out_features=10)

    def forward(self, t):
        # conv 1
        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        # conv 2
        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        # fc1
        t = t.reshape(-1, 12*4*4)
        t = self.fc1(t)
        t = F.relu(t)

        # fc2
        t = self.fc2(t)
        t = F.relu(t)

        # output
        t = self.out(t)
        # don't need softmax here since we'll use cross-entropy as activation.

        return t

RANDOM_SEED = 1
BATCH_SIZE = 100
NUM_EPOCHS = 2
#If the GPU is available use it for the computation otherwise use the CPU
#DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
DEVICE = torch.device('cpu')

train_dataset = datasets.FashionMNIST(
root = './data/FashionMNIST',
download = True,
train = True,
transform = transforms.Compose([transforms.ToTensor()])
)

test_dataset = datasets.FashionMNIST(
root = './data/FashionMNIST',
download = True,
train = False,
transform = transforms.Compose([transforms.ToTensor()])
)
train_loader = DataLoader(dataset = train_dataset, batch_size = BATCH_SIZE, shuffle= True)
test_loader = DataLoader(dataset = test_dataset, batch_size = BATCH_SIZE, shuffle= False)
torch.manual_seed(RANDOM_SEED)
model = CNN()

#model = model.to(DEVICE, memory_format=torch.channels_last)
model = model.to(DEVICE)

optimizer = torch.optim.SGD(model.parameters(), lr = 0.1)
def compute_loss (net, data_loader):
    curr_loss = 0.
    with torch.no_grad(): #disabled gradient, do not build computation graph as we are computing only loss no backward
        #Iterrating over dataloader, compute loss and add it up (instead of computing on whole dataset)
        for cnt, (features, targets) in enumerate (data_loader):
            features = features.to(DEVICE)
            targets = targets.to(DEVICE)
            #logits, probas = net(features)
            logits = net(features)
            #loss = F.nll_loss(torch.log(probas), targets)
            #for more numerically stable
            loss = F.cross_entropy(logits, targets)
            curr_loss += loss
        return float(curr_loss)/cnt



start_time = time.time()
minibatch_cost = []
epoch_cost = []
for epoch in range(NUM_EPOCHS):
    model.train()
    #model, optimizer = ipex.optimize(model, optimizer=optimizer)
    for batch_idx, (features, targets) in enumerate(train_loader):

        features = features.to(DEVICE)
        targets = targets.to(DEVICE)

        #Forward and Back Prop
        #logits, probas = model(features)
        logits = model(features) #call forward

        #cost = F.nll_loss(torch.log(probas), targets)
        cost = F.cross_entropy(logits, targets) #corss entropy does log softmax
        optimizer.zero_grad() #set gradients from prev round to 0

        cost.backward() #call backward for back prop

        #Update model parameters
        optimizer.step()

        #Logging
        minibatch_cost.append(cost.item())
        if not batch_idx % 50:
            print('Epoch : %03d/%03d | Batch %03d/%03d | Cost: %.4f'
                   %(epoch+1, NUM_EPOCHS, batch_idx, len(train_loader), cost.item()))

    cost = compute_loss(model,train_loader)
    epoch_cost.append(cost)
    print('Epoch : %03d/%03d | Cost: %.4f' % (epoch+1, NUM_EPOCHS, cost))
    print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))

print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))


def compute_accuracy(net, data_loader):
    correct_pred, num_examples = 0, 0
    with torch.no_grad():
        for features, targets in data_loader:
            features = features.to(DEVICE)
            targets = targets.to(DEVICE)
            logits = net.forward(features)
            predicted_labels = torch.argmax(logits, 1)
            num_examples += targets.size(0)
            correct_pred += (predicted_labels == targets).sum()
        return correct_pred.float() / num_examples * 100


print('Training Accuracy: %.2f' % compute_accuracy(model, train_loader))
print('Test Accuracy: %.2f' % compute_accuracy(model, test_loader))

print(model.conv1.weight.shape)
print(model.conv2.weight.shape)
print(model.fc1.weight.shape)
print(model.fc2.weight.shape)
print(model.out.weight.shape)

torch.save(model, "../output/saved_cnn_model.pt")