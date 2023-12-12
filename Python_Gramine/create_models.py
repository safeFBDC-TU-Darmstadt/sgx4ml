import os.path

from models.cnn_new import CNN_NEW
from models.CNN_FashionMNIST_model import CNN
from models.MLP_FashionMNIST_model import MLP
from models.Large_MLP2 import Large_MLP2
from models.Large_MLP1 import Large_MLP1
from models.ALEXNET_model import AlexNet
from models.ALEXNET_simple import AlexNet_simple
import torch

if not os.path.exists('output'):
    os.mkdir('output')
model = CNN_NEW()
torch.save(model, "output/saved_cnn_new_model.pt")

model = CNN()
torch.save(model, "output/saved_cnn_model.pt")

model = MLP(num_features=28 * 28, num_hidden=100, num_classes=10)
torch.save(model, "output/saved_mlp_model.pt")

model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)
model.eval()
torch.save(model, "output/saved_vgg16_model.pt")

model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg19', pretrained=True)
model.eval()
torch.save(model, "output/saved_vgg19_model.pt")

model = Large_MLP1()
torch.save(model, "output/saved_large_mlp1_model.pt")

model = Large_MLP2()
torch.save(model, "output/saved_large_mlp2_model.pt")

model = AlexNet()
torch.save(model, "output/saved_alexnet_model.pt")

model = AlexNet_simple()
torch.save(model,"output/saved_alexnet_simple.pt")
