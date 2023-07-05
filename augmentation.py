import torchvision.transforms as transforms
import torch


def AMI_augmentation_1():

  test_transform = transforms.Compose([
        #transforms.Grayscale(),
        transforms.Resize((206, 144)),
        transforms.TenCrop((176, 123)),
        transforms.Lambda(lambda crops: torch.stack(
            [transforms.ToTensor()(crop) for crop in crops])),
        transforms.Lambda(lambda tensors: torch.stack(
            [transforms.Normalize(mean=[0.3499, 0.2311, 0.1859], std=[0.1888, 0.1259, 0.1082])(t) for t in tensors])),
    ])
   
  train_transform = transforms.Compose([
      #transforms.Grayscale(),
      transforms.RandomResizedCrop((206, 144), scale=(0.8, 1.2)),
      transforms.RandomApply([transforms.ColorJitter(
          brightness=0.5, contrast=0.5, saturation=0.5)], p=0.5),
      transforms.RandomApply(
          [transforms.RandomAffine(0, translate=(0.2, 0.2))], p=0.5),
      transforms.RandomHorizontalFlip(),
      transforms.RandomApply([transforms.RandomRotation(10)], p=0.5),
      transforms.FiveCrop((176, 123)),
      transforms.Lambda(lambda crops: torch.stack(
          [transforms.ToTensor()(crop) for crop in crops])),
      transforms.Lambda(lambda tensors: torch.stack(
          [transforms.Normalize(mean=[0.3499, 0.2311, 0.1859], std=[0.1888, 0.1259, 0.1082])(t) for t in tensors])),
      transforms.Lambda(lambda tensors: torch.stack(
          [transforms.RandomErasing()(t) for t in tensors])),
  ])

  return train_transform, test_transform



def AMI_augmentation_2():

  test_transform = transforms.Compose([
      # transforms.Scale(52),
      transforms.Resize((206, 144)),
      transforms.TenCrop((176, 123)),
      transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
      transforms.Lambda(lambda tensors: torch.stack([transforms.Normalize(mean=[0.3499, 0.2311, 0.1859], std=[0.1888, 0.1259, 0.1082])(t) for t in tensors])),
  ])



  train_transform = transforms.Compose([
      transforms.RandomResizedCrop((206, 144), scale=(0.8, 1.2)),
      transforms.RandomApply([transforms.ColorJitter(
          brightness=0.5, contrast=0.5, saturation=0.5)], p=0.5),
      transforms.RandomApply([transforms.RandomAffine(0, translate=(0.2, 0.2))], p=0.5),
      transforms.RandomHorizontalFlip(),
      transforms.RandomApply([transforms.RandomRotation(10)], p=0.5),

      transforms.TenCrop((176, 123)),
      transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
      transforms.Lambda(lambda tensors: torch.stack([transforms.Normalize(mean=[0.3499, 0.2311, 0.1859], std=[0.1888, 0.1259, 0.1082])(t) for t in tensors])),
      transforms.Lambda(lambda tensors: torch.stack([transforms.RandomErasing(p=0.5)(t) for t in tensors])),
  ])

  return train_transform, test_transform