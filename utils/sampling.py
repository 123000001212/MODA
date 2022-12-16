import random

import numpy as np
from torch.utils.data import Dataset


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)
    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


def iid_sampling(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def adv_sampling(user_dataset_idx, scale):
    num_items = int(len(user_dataset_idx[0])*scale)
    adv_dict = {}
    for i in range(len(user_dataset_idx)-1):
        adv_dict = list(set(adv_dict) | set(random.sample(user_dataset_idx[i], num_items)))
    return adv_dict
