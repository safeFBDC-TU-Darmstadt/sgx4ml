# Load MLP model
import random
import time

import torch
from torchvision import datasets
from torchvision.transforms import ToTensor

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


class MLP(torch.nn.Module):

    def __init__(self, num_features, num_hidden, num_classes):
        super().__init__()

        self.num_classes = num_classes

        ### 1st hidden layer
        self.linear_1 = torch.nn.Linear(num_features, num_hidden)
        self.linear_1.weight.detach().normal_(0.0, 0.1)
        self.linear_1.bias.detach().zero_()

        ### Output layer
        self.linear_out = torch.nn.Linear(num_hidden, num_classes)
        self.linear_out.weight.detach().normal_(0.0, 0.1)
        self.linear_out.bias.detach().zero_()

    def forward(self, x):
        out = self.linear_1(x)
        out = torch.sigmoid(out)
        logits = self.linear_out(out)
        # probas = torch.softmax(logits, dim = 1)
        return logits  # , probas


model = torch.load("output/saved_mlp_model.pt")
model.eval()

# Pick random item from test set
test_data = datasets.FashionMNIST(
    root="./data",
    train=False,
    download=True,
    transform=ToTensor()
)
item_num = random.randint(0, len(test_data))
img, label = test_data[item_num]
print("This item is:", labels_map[label])
input = img.reshape(-1,28*28)

start_time = time.time()
output = model(input)
end_time = time.time()

print("Guess by model:",labels_map[int(torch.argmax(output))])
print("Time to infer:", (end_time-start_time))