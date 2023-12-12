from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


def get_training_data():
    return datasets.FashionMNIST(
        root="./data",
        train=True,
        download=True,
        transform=ToTensor()
    )


def get_training_dataloader(batch_size, bool):
    return DataLoader(dataset=get_training_data(), batch_size=batch_size, shuffle=bool)


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
