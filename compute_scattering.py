"""
Author: ta
Date and time: 20/11/2018 - 22:59
"""

import os

os.environ["KYMATIO_BACKEND_2D"] = "skcuda"

from kymatio import Scattering2D
import kymatio.scattering2d.backend as backend

print('[*] Backend for Scattering2D: {}'.format(backend.NAME))

from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from ImageDataset import ImageDataset


def compute_scattering(dir_images, dir_to_save, scattering, batch_size):
    dir_to_save.mkdir()

    dataset = ImageDataset(dir_images)
    dataloader = DataLoader(dataset, batch_size, pin_memory=True, num_workers=1)

    for idx_batch, current_batch in enumerate(tqdm(dataloader)):
        images = current_batch.float().cuda()
        if scattering is not None:
            s_images = scattering(images).cpu().numpy()
            s_images = np.reshape(s_images, (batch_size, -1, s_images.shape[-1], s_images.shape[-1]))
        else:
            s_images = images.cpu().numpy()
        for idx_local in range(batch_size):
            idx_global = idx_local + idx_batch * batch_size
            filename = dir_to_save / '{}.npy'.format(idx_global)
            temp = s_images[idx_local]
            np.save(filename, temp)


def main():
    dir_datasets = Path('~/datasets/').expanduser()
    dataset_attribute = '128_rgb_512_512'
    dir_dataset = dir_datasets / 'celeba' / dataset_attribute

    batch_size = 512

    for J in range(4, 5):
        dir_to_save = dir_datasets / 'celeba' / '{}_SJ{}'.format(dataset_attribute, J)
        dir_to_save.mkdir()

        if J == 0:
            scattering = None
        else:
            scattering = Scattering2D(J, (128, 128))
            scattering.cuda()

        compute_scattering(dir_dataset / 'train', dir_to_save / 'train', scattering, batch_size)
        compute_scattering(dir_dataset / 'test', dir_to_save / 'test', scattering, batch_size)


if __name__ == '__main__':
    main()
