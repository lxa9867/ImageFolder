from torchvision.datasets import ImageFolder


class ImageFolderTwoTransform(ImageFolder):
    def __init__(self, root, transform1=None, transform2=None, target_transform=None, **kwargs):
        super(ImageFolderTwoTransform, self).__init__(root, transform=transform1, target_transform=target_transform, **kwargs)
        self.transform2 = transform2
    
    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        sample_ = sample.copy()
        
        if self.transform is not None:
            sample = self.transform(sample)
        if self.transform2 is not None:
            sample_ = self.transform2(sample_)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, sample_, target