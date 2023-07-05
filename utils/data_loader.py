import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.utils.data as data_utils
#from utils.fashion_mnist import MNIST, FashionMNIST
from utils.motor import getMotorDataLoader

def get_data_loader(args):

    if args.dataset == 'motor':
      if args.large_size:
        trans = transforms.Compose([
            transforms.Resize(128),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
      else:
        trans = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
      
      
      train_dataloader = getMotorDataLoader(args.train_dataroot, batch_size=args.batch_size, class_name=args.class_name, transform=trans)
      test_dataloader = getMotorDataLoader(args.test_dataroot, batch_size=args.batch_size, class_name=args.class_name, transform=trans)
    
    return train_dataloader, test_dataloader
