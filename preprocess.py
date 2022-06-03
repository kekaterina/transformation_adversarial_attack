from torch.utils.data import Dataset, DataLoader
from torch import Tensor, LongTensor
import numpy as np


class ImagenetDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        return x, y

    def __len__(self):
        return len(self.data)


def get_dict_renumber_classes(y_train):
    labels_mapping = {}
    for i, e in enumerate(set(y_train)):
        labels_mapping[e] = i
    return labels_mapping


def renumber_classes(dict_new_classes, classes):
    renumber_y_train = np.zeros(classes.shape)
    for i, e in enumerate(classes):
        renumber_y_train[i] = dict_new_classes[e]
    return renumber_y_train


def preprocess_data_short_cbn(image_arr_path, lab_arr_path, renumber=True):
    images = np.load(image_arr_path)
    labels = np.load(lab_arr_path)
    if renumber:
        dict_new_classes = get_dict_renumber_classes(labels)
        labels = renumber_classes(dict_new_classes, labels)
    return images, labels


def get_dataloader(x, y, device, shuffle=True, batch_size=10):
    x = Tensor(x).to(device)
    y = LongTensor(y).to(device)
    dataset = ImagenetDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataloader


def get_target_label(model, x, acceptable_labels, top=5):
    """
    This function choice target label for target attack.
    The function selects a label only from those that are in the training
    dataset (there are not 1000 classes in the dataset used), and the function
    also tries to choose the most distant label from the true label.
    """
    tr = model(x)
    tr_set = np.argsort(tr.cpu().detach().numpy())[0][-top:]
    mask = np.in1d(acceptable_labels, tr_set)
    accept_lab_here = acceptable_labels[~mask]

    if len(accept_lab_here) == 0:
        y = LongTensor(tr_set[0])
    else:
        y = LongTensor(np.random.choice(accept_lab_here, 1))
    return y
