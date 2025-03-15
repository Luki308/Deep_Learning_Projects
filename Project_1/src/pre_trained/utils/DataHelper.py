import os
from enum import Enum

import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

cinic_directory = "C:\\Users\\agata\\.cache\\kagglehub\\datasets\\mengcius\\cinic10\\versions\\1"
cinic_mean = [0.47889522, 0.47227842, 0.43047404]
cinic_std = [0.24205776, 0.23828046, 0.25874835]


class SimpleAugmentation(Enum):
    RandomHorizontalFlip = 1
    RandomVerticalFlip = 2
    RandomRotation = 3
    ColorJitter = 4
    RandomErasing = 5


class DataHelper:
    augmentation: list[SimpleAugmentation]
    resize: int
    batch_size: int
    subset_fraction: float

    def __init__(self, resize=None, batch_size=64, subset_fraction=None, augmentations=[]):
        """
        :param resize: Decides image size
        :param batch_size: Batch size for training, validation, and testing
        :param subset_fraction: Fraction of dataset to use
        :param augmentations: Decides augmentations to apply
        """
        self.batch_size = batch_size
        self.subset_fraction = subset_fraction
        self.img_size = resize if resize is not None else 32  # default to no resizing

        self.test_transform, self.train_transform = self._get_transforms(augmentations)

    def _get_transforms(self, augmentations):
        """
        Prepares transformations for training and testing including augmentations.
        :param augmentations: List of SimpleAugmentation enums defining augmentations to apply
        :return: train_transform, test_transform
        """
        transformations = []
        for augmentation in augmentations:
            match augmentation:
                case SimpleAugmentation.RandomHorizontalFlip:
                    transformations.append(transforms.RandomHorizontalFlip())
                case SimpleAugmentation.RandomVerticalFlip:
                    transformations.append(transforms.RandomVerticalFlip())
                case SimpleAugmentation.RandomRotation:
                    transformations.append(transforms.RandomRotation(30))
                case SimpleAugmentation.ColorJitter:
                    transformations.append(
                        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2))
                case SimpleAugmentation.RandomErasing:
                    pass

        transformations.extend([transforms.ToTensor(), transforms.Normalize(mean=cinic_mean, std=cinic_std),
                                transforms.Resize((self.img_size, self.img_size),
                                                  interpolation=transforms.InterpolationMode.BICUBIC)])

        if SimpleAugmentation.RandomErasing in augmentations:
            transformations.append(transforms.RandomErasing())

        train_transform = transforms.Compose(transformations)

        test_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=cinic_mean, std=cinic_std),
             transforms.Resize((self.img_size, self.img_size),
                               interpolation=transforms.InterpolationMode.BICUBIC), ])

        return train_transform, test_transform

    def get_loaders(self):
        """
        Load CINIC-10 dataset and return DataLoader objects. applying transformations
        """
        train_dataset = torchvision.datasets.ImageFolder(os.path.join(cinic_directory, "train"),
                                                         transform=self.train_transform)
        val_dataset = torchvision.datasets.ImageFolder(os.path.join(cinic_directory, "valid"),
                                                       transform=self.test_transform)
        test_dataset = torchvision.datasets.ImageFolder(os.path.join(cinic_directory, "test"),
                                                        transform=self.test_transform)

        if self.subset_fraction:
            def stratified_subset(dataset, subset_fraction):
                """Ensures class balance in the subset."""
                targets = [label for _, label in dataset.imgs]
                train_indices, _ = train_test_split(range(len(targets)), train_size=subset_fraction, stratify=targets,
                                                    random_state=3371)
                return Subset(dataset, train_indices)

            train_dataset = stratified_subset(train_dataset, self.subset_fraction)
            val_dataset = stratified_subset(val_dataset, self.subset_fraction)
            test_dataset = stratified_subset(test_dataset, self.subset_fraction)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4,
                                  pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4,
                                 pin_memory=True)

        return train_loader, val_loader, test_loader
