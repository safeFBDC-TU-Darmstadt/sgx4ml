import torch
model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)
model.eval()

torch.save(model, "../output/saved_vgg16_model.pt")