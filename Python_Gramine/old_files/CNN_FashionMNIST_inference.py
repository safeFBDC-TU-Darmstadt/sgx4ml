import time

import torch.nn.functional as F
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
import random

labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}


class CNN(torch.nn.Module):

    def __init__(self):
        super().__init__()

        # define layers
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)

        self.fc1 = torch.nn.Linear(in_features=12 * 4 * 4, out_features=120)
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
        t = t.reshape(-1, 12 * 4 * 4)
        t = self.fc1(t)
        t = F.relu(t)

        # fc2
        t = self.fc2(t)
        t = F.relu(t)

        # output
        t = self.out(t)
        # don't need softmax here since we'll use cross-entropy as activation.

        return t


# Load CNN model
model2 = torch.load("output/saved_cnn_model.pt")
model2.eval()
#model2.to(DEVICE)

# Pick random item from test set
test_data = datasets.FashionMNIST(
    root="./data",
    train=False,
    download=True,
    transform=ToTensor()
)
item_num = random.randint(0, len(test_data))
img, label = test_data[item_num]
#img = img.to(DEVICE)
print("This item is:", labels_map[label])

# model inference
start_time = time.time()
output = model2(img)
end_time = time.time()

print("Guess by model:", labels_map[int(torch.argmax(output))])
print("Time to infer:", (end_time - start_time))
