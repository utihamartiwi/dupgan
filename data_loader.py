import torch
from torchvision import datasets
from torchvision import transforms

def get_loader(config):
    transform = transforms.Compose([
                    transforms.Scale(config.image_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    svhn = datasets.SVHN(root=config.svhn_path, download=True, transform=transform)
    mnist = datasets.MNIST(root=config.mnist_path, download=True, transform=transform)
    mnist_val = datasets.MNIST(root=config.mnist_path, train=False, download=True, transform=transform)
    
    svhn_loader = torch.utils.data.DataLoader(dataset=svhn,
                                              batch_size=config.batch_size,
                                              shuffle=True,
                                              num_workers=config.num_workers)

    mnist_loader = torch.utils.data.DataLoader(dataset=mnist,
                                               batch_size=config.batch_size,
                                               shuffle=False,
                                               num_workers=config.num_workers)
    mnist_val_loader = torch.utils.data.DataLoader(dataset=mnist_val,
                                               batch_size=100,
                                               shuffle=False,
                                               num_workers=config.num_workers)
    return svhn_loader, mnist_loader, mnist_val_loader
