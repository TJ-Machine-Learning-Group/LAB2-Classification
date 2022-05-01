import config
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets


def prepare_cifar10():
    transform = transforms.Compose([
        #transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    train_dataset = datasets.CIFAR10(config.cifar10_train_path, train=True, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_dataset = datasets.CIFAR10(config.cifar10_test_path, train=False, transform=transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True)

    return train_loader, len(train_dataset), test_loader, len(test_dataset)