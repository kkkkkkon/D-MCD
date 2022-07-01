import torch
import torchvision
from torchvision import transforms
# from torchvision.datasets import ImageFolderWithPath
from imagewithpath import ImageFolderWithPath
from torch.utils.data import DataLoader
from torchtoolbox.transform import Cutout
import random
import numpy as np

def get_transforms(train_path, val_path):
    data_transforms = {
        'source': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'target': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'validation': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    return data_transforms

# def _init_fn(worker_id): 
#     random.seed(10 + worker_id)
#     np.random.seed(10 + worker_id)
#     torch.manual_seed(10 + worker_id)
#     torch.cuda.manual_seed(10 + worker_id)
#     torch.cuda.manual_seed_all(10 + worker_id)

def load_data(train_path, val_path, batch_size, random_shuffle=True, drop_last=True):

    data_transforms = get_transforms(train_path, val_path)
    
    source_dataset = ImageFolderWithPath(
        root=train_path, 
        transform=data_transforms['source'],
    )
    target_dataset = ImageFolderWithPath(
        root=val_path,
        transform=data_transforms['target'],
    )
    validation_dataset = ImageFolderWithPath(
        root=val_path,
        transform=data_transforms['validation']
    )
    source_loader = DataLoader(
        dataset=source_dataset, 
        batch_size=batch_size,
        shuffle=random_shuffle,
        drop_last=drop_last,
        num_workers=16,
        # worker_init_fn=_init_fn
    )
    target_loader = DataLoader(
        dataset=target_dataset, 
        batch_size=batch_size,
        shuffle=random_shuffle,
        drop_last=drop_last,
        num_workers=16,
        # # worker_init_fn=_init_fn
    )
    validation_loader = DataLoader(
        dataset=validation_dataset, 
        batch_size=batch_size,
        shuffle=random_shuffle,
        drop_last=False,
        num_workers=16,
        # # worker_init_fn=_init_fn
    )
    return source_loader, target_loader, validation_loader