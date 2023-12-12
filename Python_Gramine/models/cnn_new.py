import torch
import torch.nn.functional as F


class CNN_NEW(torch.nn.Module):

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
        t = t.reshape(-1, 6 * 24 * 24)
        t = t[:, 0:6 * 12 * 12]
        t = t.reshape(-1, 6, 12, 12)
        # t = F.max_pool2d(t, kernel_size=2, stride=2)
        # print(t.size())
        # x = t.clone()
        # x = t.reshape(-1, 6 * 12 * 12)
        # x = x[0:6 * 4 * 4]
        # x = x.reshape(1, 6, 4, 4)
        # print(x.size())

        # conv 2
        t = self.conv2(t)
        t = F.relu(t)
        t = t.reshape(-1, 12 * 8 * 8)
        t = t[:, 0:12 * 4 * 4]
        t = t.reshape(-1, 12, 4, 4)
        # t = F.max_pool2d(t, kernel_size=2, stride=2)

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

# model = CNN_NEW()
# torch.save(model,"output/saved_cnn_new_model.pt")
