import torch
import torch.nn.functional as F


class Large_MLP2(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.linear_0 = torch.nn.Linear(28 * 28, 4000)

        self.linear_1 = torch.nn.Linear(4000, 2000)

        self.linear_2 = torch.nn.Linear(2000, 1000)

        self.linear_3 = torch.nn.Linear(1000, 10)

        self.relu = torch.nn.ReLU()

    def forward(self, x):
        out = self.linear_0(x)
        out = self.relu(out)

        out = self.linear_1(out)
        out = self.relu(out)

        out = self.linear_2(out)
        out = self.relu(out)

        out = self.linear_3(out)
        out = torch.sigmoid(out)

        # logits = self.linear_out(out)
        # probas = torch.softmax(logits, dim = 1)
        return out  # , probas

# model = Large_MLP2()
# torch.save(model,"output/saved_large_mlp2_model.pt")
