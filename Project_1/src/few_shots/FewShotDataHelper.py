import random

import torch
import intel_extension_for_pytorch as ipex
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset

torch.manual_seed(0)
random.seed(0)

cinic_directory = "Project_1/data"
cinic_mean = [0.47889522, 0.47227842, 0.43047404]
cinic_std = [0.24205776, 0.23828046, 0.25874835]


def stratified_subset(dataset, subset_fraction):
    targets = [label for _, label in dataset.imgs]
    train_indices, _ = train_test_split(range(len(targets)), train_size=subset_fraction, stratify=targets,
                                        random_state=3371)
    return Subset(dataset, train_indices)


def build_class_index(dataset):
    class_to_indices = {}
    for idx, (_, label) in enumerate(dataset):
        if label not in class_to_indices:
            class_to_indices[label] = []
        class_to_indices[label].append(idx)
    return class_to_indices


class FewShotDataset(Dataset):
    def __init__(self, dataset, class_to_indices, n_way=5, k_shot=5, query_size=15, num_tasks=10000):
        self.dataset = dataset
        self.class_to_indices = class_to_indices
        self.n_way = n_way
        self.k_shot = k_shot
        self.query_size = query_size
        self.num_tasks = num_tasks

    def __len__(self):
        return self.num_tasks

    def __getitem__(self, idx):
        selected_classes = random.sample(list(self.class_to_indices.keys()), self.n_way)
        support_images, support_labels = [], []
        query_images, query_labels = [], []
        for new_label, cls in enumerate(selected_classes):
            indices = random.sample(self.class_to_indices[cls], self.k_shot + self.query_size)
            support_idx = indices[:self.k_shot]
            query_idx = indices[self.k_shot:]
            for i in support_idx:
                image, _ = self.dataset[i]
                support_images.append(image)
                support_labels.append(new_label)
            for i in query_idx:
                image, _ = self.dataset[i]
                query_images.append(image)
                query_labels.append(new_label)
        # stack imgs
        x_support = torch.stack(support_images)
        y_support = torch.tensor(support_labels, dtype=torch.long)
        x_query = torch.stack(query_images)
        y_query = torch.tensor(query_labels, dtype=torch.long)
        return (x_support, y_support), (x_query, y_query)