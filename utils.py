# coding=utf-8

"""
Author: angles
Date and time: 27/04/18 - 17:58
"""

import os
from datetime import datetime

import numpy as np
from PIL import Image


def load_image(filename):
    return np.ascontiguousarray(Image.open(filename), dtype=np.uint8)


def normalize(vector):
    norm = np.sqrt(np.sum(vector ** 2))
    return vector / norm


def get_nb_files(input_dir):
    list_files = [file for file in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, file))]
    return len(list_files)


def create_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def create_name_experiment(parameters, attribute_experiment):
    name_experiment = '{}_{}_{}_ncfl{}_{}'.format(parameters['dataset'],
                                                  parameters['dataset_attribute'],
                                                  parameters['embedding_attribute'],
                                                  parameters['nb_channels_first_layer'],
                                                  attribute_experiment)

    print('Name experiment: {}'.format(name_experiment))

    return name_experiment


class AverageMeter(object):
    """
    Computes and stores the average and current value
    """

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def now():
    return datetime.now().strftime("%d%m%Y%H%M%S")


def get_hms(seconds):
    seconds = int(seconds)
    minutes = seconds // 60
    rseconds = seconds - 60 * minutes
    return '{}m{}s'.format(minutes, rseconds)
