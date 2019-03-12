"""
Author: ta
Date and time: 20/11/2018 - 23:56
"""

from torch.utils.data import Dataset

from utils import get_nb_files, load_image


class ImageDataset(Dataset):
    def __init__(self, dir_x, nb_channels=3):
        assert nb_channels in [1, 3]
        self.nb_channels = nb_channels
        self.nb_files = get_nb_files(dir_x)
        self.dir_x = dir_x

    def __len__(self):
        return self.nb_files

    def __getitem__(self, idx):
        filename = self.dir_x / '{}.png'.format(idx)
        if self.nb_channels == 3:
            x = load_image(filename).transpose((2, 0, 1)) / 255.0
        else:
            x = load_image(filename) / 255.0

        return x
