import os
import torch 
import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import webdataset as wds
import pickle
from glob import glob 
from torch.nn.utils.rnn import pad_sequence
from .list_dataset import ImageListDataset
from .folder_dataset import ImageFolderTwoTransform

class ToNumpy:
    def __call__(self, pil_img):
        np_img = np.array(pil_img, dtype=np.uint8)
        if np_img.ndim < 3:
            np_img = np.expand_dims(np_img, axis=-1)
        np_img = np.rollaxis(np_img, 2)  # HWC to CHW
        return np_img


def create_transforms(image_size, is_train=True, use_prefetcher=False, 
                      mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    
    transform_list = [
        transforms.Resize(int(image_size * 1.125)),
    ]
    if is_train:
        transform_list.extend([
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
        ])
    else:
        transform_list.extend([
            transforms.CenterCrop(image_size),
        ])
    if use_prefetcher:
        transform_list.append(ToNumpy())
    else:
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize(mean=mean, std=std))
    return transforms.Compose(transform_list)


def decode_pkl(data):
    return pickle.loads(data['.pkl'])
    


def create_dataset(dataset_name, data_dir, image_size, is_train=True, use_prefetcher=False):
    
    if dataset_name == "imagenet":
        
        dataset = ImageFolder(data_dir, transform=create_transforms(image_size, is_train, use_prefetcher))
        # dataset = ImageFolderTwoTransform(data_dir, 
        #                                   transform1=create_transforms(image_size, is_train, use_prefetcher, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), 
        #                                   transform2=create_transforms(image_size, is_train, use_prefetcher, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
    
    
    elif dataset_name == 'imagenet_list':
        
        data_list = sorted(glob(os.path.join(data_dir, "*/*.JPEG")))
        dataset = ImageListDataset(data_list, transform=create_transforms(image_size, is_train, use_prefetcher))
    
    
    elif dataset_name == 'imagenet_webdataset':
        
        if ':' in data_dir:
            data_dir_list = data_dir.split(':')
        else:
            data_dir_list = [data_dir]
        urls = []
        for data_dir in data_dir_list:        
            urls.extend(sorted(glob(os.path.join(data_dir, "*.tar"))))
        
        dataset = wds.DataPipeline(
            wds.SimpleShardList(urls),

            # at this point we have an iterator over all the shards
            wds.shuffle(1000),

            # add wds.split_by_node here if you are using multiple nodes
            wds.split_by_worker,

            # at this point, we have an iterator over the shards assigned to each worker
            wds.tarfile_to_samples(),

            # this shuffles the samples in memory
            wds.shuffle(5000),

            # this decodes the images and json
            # wds.decode("pkl"),
            wds.map(decode_pkl),
        )

    return dataset